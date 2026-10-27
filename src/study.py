"""
study.py
========
Study-level helpers: panel construction, strategy weight schedules, scenario runs,
random direct-equity portfolios, table/figure export.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import etf_engine as E

# ----------------------------------------------------------------------------
# Plot style: journal-ready, colour-blind-validated categorical order
# ----------------------------------------------------------------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STOCK_GREY = "#8a8984"


def set_style():
    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42})


# ----------------------------------------------------------------------------
# Output registry
# ----------------------------------------------------------------------------
class Outputs:
    """Saves tables (CSV + LaTeX + Excel workbook) and figures (PNG + PDF) with an index."""

    def __init__(self, dirs: Dict[str, str]):
        self.dirs = dirs
        self.tables: Dict[str, pd.DataFrame] = {}
        self.index: List[Dict] = []

    def table(self, df: pd.DataFrame, name: str, caption: str, index: bool = True, digits: int = 4):
        df = df.copy()
        df.to_csv(os.path.join(self.dirs["tables"], f"{name}.csv"), index=index)
        try:
            tex = df.to_latex(index=index, float_format=lambda x: f"{x:.{digits}f}",
                              caption=caption, label=f"tab:{name}", escape=True, longtable=False)
            with open(os.path.join(self.dirs["tables"], f"{name}.tex"), "w") as f:
                f.write(tex)
        except Exception as e:  # pragma: no cover
            print("LaTeX export skipped:", e)
        self.tables[name] = df if index else df.set_index(df.columns[0])
        self.index.append({"type": "table", "name": name, "caption": caption})

    def figure(self, fig, name: str, caption: str):
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.dirs["figures"], f"{name}.{ext}"), bbox_inches="tight")
        self.index.append({"type": "figure", "name": name, "caption": caption})

    def write_workbook(self):
        path = os.path.join(self.dirs["tables"], "all_tables.xlsx")
        with pd.ExcelWriter(path) as xw:
            pd.DataFrame(self.index).to_excel(xw, sheet_name="index", index=False)
            for n, df in self.tables.items():
                df.to_excel(xw, sheet_name=n[:31])
        pd.DataFrame(self.index).to_csv(os.path.join(self.dirs["results"], "output_index.csv"), index=False)
        return path


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------
def wide_monthly(long: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    w = long.pivot_table(index="date", columns=key, values=value, aggfunc="last").sort_index()
    return w.resample("ME").last()


def wide_daily(long: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    return long.pivot_table(index="date", columns=key, values=value, aggfunc="last").sort_index()


def contributions(dates: pd.DatetimeIndex, mode: str, amount: float) -> pd.Series:
    s = pd.Series(0.0, index=dates)
    if mode == "lump":
        s.iloc[0] = amount
    else:
        s[:] = amount
    return s


# ----------------------------------------------------------------------------
# ETF weight schedules (computed once, shared by every investor scenario)
# ----------------------------------------------------------------------------
def weight_schedule(strategy: str, cov_method: str, Rm: pd.DataFrame, Rd: pd.DataFrame,
                    rf: pd.Series, oos_dates: pd.DatetimeIndex, rebal: set, window: int,
                    cap: float, single_asset: str = None, equity_assets: List[str] = None) -> pd.DataFrame:
    """Target weights at every rebalance date (and the first OOS date), estimated from data
    available up to that date only (rolling window of `window` months)."""
    assets = list(Rm.columns)
    rows = {}
    for d in oos_dates:
        if d != oos_dates[0] and d not in rebal:
            continue
        if strategy == "EQ100":
            w = pd.Series(0.0, index=assets); w[single_asset] = 1.0
        elif strategy == "EQ_EW":
            w = pd.Series(0.0, index=assets); w[equity_assets] = 1.0 / len(equity_assets)
        elif strategy == "EW":
            w = E.w_equal(assets)
        else:
            win = Rm.loc[:d].iloc[-window:]
            if cov_method == "sample":
                cov = E.cov_sample(win)
            elif cov_method == "lw":
                cov = E.cov_ledoit_wolf(win)
            elif cov_method == "dcc":
                dwin = Rd.loc[:d].iloc[-window * 21:]
                try:
                    cov = E.dcc_forecast(dwin)
                except Exception:
                    cov = E.cov_ledoit_wolf(win)
            else:
                raise ValueError(cov_method)
            if strategy == "MinVar":
                w = E.w_minvar(cov, assets, cap)
            elif strategy == "MaxSharpe":
                w = E.w_maxsharpe(win.mean().values, cov, float(rf.loc[:d].iloc[-window:].mean()), assets, cap)
            elif strategy == "RiskParity":
                w = E.w_riskparity(cov, assets)
            else:
                raise ValueError(strategy)
            w.attrs["dr"] = E.diversification_ratio(w, cov)
        rows[d] = w
    return pd.DataFrame(rows).T


def make_weight_fn(sched: pd.DataFrame, dates: pd.DatetimeIndex):
    def fn(t):
        d = dates[t]
        s = sched.loc[:d]
        return s.iloc[-1] if len(s) else sched.iloc[0]
    return fn


def scenario_objects(mode: str, broker: str, slab: float, cfg: dict, regime: str = None,
                     extra_bps: float = 0.0, gf_prices=None):
    cc = cfg["costs"]
    cost = E.CostModel(brokerage=broker, flat_fee=cc["flat_fee"],
                       stt={k: tuple(v) for k, v in cc["stt"].items()},
                       exchange=cc["exchange"], sebi=cc["sebi"], stamp_buy=cc["stamp_buy"],
                       gst=cc["gst"], dp_charge_per_sell=cc["dp_charge_per_sell"],
                       half_spread=cc["half_spread"], extra_bps=extra_bps,
                       frictionless=(mode == "frictionless"))
    tx = cfg["tax"]
    tax = E.TaxEngine(regime="none" if mode != "costs_tax" else (regime or tx["regime"]),
                      slab=slab, cess=tx["cess"], eq_st_rate=tx["eq_st_rate"],
                      eq_lt_rate=tx["eq_lt_rate"], eq_lt_exemption=tx["eq_lt_exemption"],
                      oth_lt_rate=tx["oth_lt_rate"], cii={int(k): v for k, v in tx["cii"].items()},
                      gf_prices=gf_prices)
    integer = mode != "frictionless"
    return cost, tax, integer


def run_one(prices, types, sched, contrib, rebal, mode, broker, slab, cfg, threshold=None,
            regime=None, extra_bps=0.0, gf_prices=None):
    cost, tax, integer = scenario_objects(mode, broker, slab, cfg, regime, extra_bps, gf_prices)
    return E.simulate(prices, types, make_weight_fn(sched, prices.index), rebal, contrib, cost, tax,
                      integer=integer, threshold=threshold, tax_month=cfg["tax"]["payment_month"],
                      sip_single_order=(broker in cfg.get("sip_single_order_brokers", [])) and mode != "frictionless")


# ----------------------------------------------------------------------------
# Random direct-equity portfolios
# ----------------------------------------------------------------------------
def draw_stock_sets(P: pd.DataFrame, elig: pd.DataFrame, start: pd.Timestamp, N: int,
                    draws: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed + N)
    pool = [c for c in elig.columns if bool(elig.loc[start, c]) and np.isfinite(P.loc[start, c])]
    return [list(rng.choice(pool, size=N, replace=False)) for _ in range(draws)]


def run_stock_portfolio(P_oos: pd.DataFrame, ids: List[str], contrib, rebal, mode, broker, cfg,
                        regime=None, gf_prices=None):
    prices = P_oos[ids]
    types = {i: "stock" for i in ids}
    cost, tax, integer = scenario_objects(mode, broker, 0.2, cfg, regime, 0.0, gf_prices)

    def wfn(t):
        live = [i for i in ids if np.isfinite(prices.iloc[t][i])]
        return pd.Series(1.0 / len(live), index=live) if live else pd.Series(dtype=float)
    return E.simulate(prices, types, wfn, rebal, contrib, cost, tax, integer=integer,
                      tax_month=cfg["tax"]["payment_month"],
                      sip_single_order=(broker in cfg.get("sip_single_order_brokers", [])) and mode != "frictionless")


# ----------------------------------------------------------------------------
# Sub-periods
# ----------------------------------------------------------------------------
def subperiod_table(returns: Dict[str, pd.Series], rf: pd.Series, periods: Dict[str, List[str]]):
    rows = []
    for pname, (a, b) in periods.items():
        for s, r in returns.items():
            rr = r.loc[a:b].dropna()
            if len(rr) < 6:
                continue
            m = E.perf_metrics(rr, rf)
            rows.append({"subperiod": pname, "strategy": s, "months": len(rr),
                         "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_cumulative(ret: Dict[str, pd.Series], band: pd.DataFrame = None, band_label: str = ""):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if band is not None and len(band):
        cum = (1 + band.fillna(0)).cumprod()
        lo, md, hi = cum.quantile(0.05, axis=1), cum.median(axis=1), cum.quantile(0.95, axis=1)
        ax.fill_between(cum.index, lo, hi, color=STOCK_GREY, alpha=0.18, lw=0,
                        label=f"{band_label}: 5th-95th pct")
        ax.plot(md.index, md, color=STOCK_GREY, lw=1.4, ls="--", label=f"{band_label}: median")
    for i, (k, r) in enumerate(ret.items()):
        c = (1 + r.fillna(0)).cumprod()
        ax.plot(c.index, c, color=PALETTE[i % 8], label=k)
    ax.set_ylabel("Growth of Rs 1 (time-weighted, net)")
    ax.legend(ncol=3, loc="upper left")
    return fig


def fig_drawdowns(ret: Dict[str, pd.Series]):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, (k, r) in enumerate(ret.items()):
        w = (1 + r.fillna(0)).cumprod()
        ax.plot(w.index, (w / w.cummax() - 1) * 100, color=PALETTE[i % 8], label=k)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(ncol=3, loc="lower left")
    return fig


def fig_corr(Rm: pd.DataFrame, labels: Dict[str, str]):
    C = Rm.corr()
    names = [labels.get(c, c) for c in C.columns]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    return fig


def fig_dcc(dcc: pd.DataFrame, pairs: List[str]):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    m = dcc[pairs].resample("ME").mean()
    for i, p in enumerate(pairs):
        ax.plot(m.index, m[p], color=PALETTE[i % 8], label=p)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("DCC correlation (monthly avg)")
    ax.legend(ncol=2, loc="lower left")
    return fig


def fig_sharpe_dist(stock_sharpes: pd.DataFrame, etf_sharpes: Dict[str, float]):
    """stock_sharpes: long df with columns N, sharpe."""
    Ns = sorted(stock_sharpes["N"].unique())
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    data = [stock_sharpes.loc[stock_sharpes["N"] == n, "sharpe"].dropna().values for n in Ns]
    parts = ax.violinplot(data, positions=range(len(Ns)), showmedians=True, widths=0.75)
    for b in parts["bodies"]:
        b.set_facecolor(STOCK_GREY); b.set_edgecolor(INK2); b.set_alpha(0.35)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[k].set_color(INK2); parts[k].set_linewidth(1)
    for i, (k, v) in enumerate(etf_sharpes.items()):
        ax.axhline(v, color=PALETTE[i % 8], lw=1.6, label=f"{k} (Sharpe {v:.2f})")
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels([f"{n} stocks" for n in Ns])
    ax.set_ylabel("Annualised Sharpe ratio (net of costs & tax)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    return fig


def fig_erosion(df: pd.DataFrame):
    """df columns: investor, portfolio, cost_pct, tax_pct (as % of contributions)."""
    inv = list(dict.fromkeys(df["investor"]))
    ports = list(dict.fromkeys(df["portfolio"]))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    w = 0.8 / len(ports)
    for j, p in enumerate(ports):
        sub = df[df["portfolio"] == p].set_index("investor").reindex(inv)
        x = np.arange(len(inv)) + (j - (len(ports) - 1) / 2) * w
        ax.bar(x, sub["cost_pct"], w * 0.92, color=PALETTE[j % 8], label=f"{p}: costs")
        ax.bar(x, sub["tax_pct"], w * 0.92, bottom=sub["cost_pct"], color=PALETTE[j % 8],
               alpha=0.45, hatch="///", edgecolor="white", linewidth=0, label=f"{p}: tax")
    ax.set_xticks(range(len(inv))); ax.set_xticklabels(inv, rotation=20, ha="right")
    ax.set_ylabel("% of total contributions")
    ax.legend(ncol=2)
    return fig


def fig_frontier(Rm: pd.DataFrame, labels: Dict[str, str], points: Dict[str, pd.Series],
                 stock_cloud: pd.DataFrame = None):
    from sklearn.covariance import LedoitWolf
    mu = Rm.mean().values * 12
    cov = LedoitWolf().fit(Rm.values).covariance_ * 12
    n = len(mu)
    from scipy.optimize import minimize
    targets = np.linspace(mu.min(), mu.max(), 40)
    vols = []
    for tg in targets:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, tg=tg: w @ mu - tg}]
        r = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), bounds=[(0, 1)] * n,
                     constraints=cons, method="SLSQP")
        vols.append(np.sqrt(r.fun) if r.success else np.nan)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if stock_cloud is not None and len(stock_cloud):
        ax.scatter(stock_cloud["ann_vol"] * 100, stock_cloud["ann_return"] * 100, s=8,
                   color=STOCK_GREY, alpha=0.35, lw=0, label="Random 10-stock portfolios (realised)")
    ax.plot(np.array(vols) * 100, targets * 100, color=INK, lw=1.6, label="Long-only ETF frontier")
    sd = np.sqrt(np.diag(cov))
    for i, c in enumerate(Rm.columns):
        ax.scatter(sd[i] * 100, mu[i] * 100, s=40, marker="s", color=INK2)
        ax.annotate(labels.get(c, c), (sd[i] * 100, mu[i] * 100), xytext=(4, 3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    for i, (k, w) in enumerate(points.items()):
        w = w.reindex(Rm.columns).fillna(0).values
        ax.scatter(np.sqrt(w @ cov @ w) * 100, w @ mu * 100, s=70, color=PALETTE[(i + 1) % 8],
                   edgecolor="white", linewidth=1.5, zorder=5, label=k)
    ax.set_xlabel("Annualised volatility (%)"); ax.set_ylabel("Annualised mean return (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5)
    return fig


def fig_rolling_sharpe(ret: Dict[str, pd.Series], rf: pd.Series, win: int = 36):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, (k, r) in enumerate(ret.items()):
        ex = r - rf.reindex(r.index).fillna(0)
        rs = ex.rolling(win).mean() / ex.rolling(win).std() * np.sqrt(12)
        ax.plot(rs.index, rs, color=PALETTE[i % 8], label=k)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(f"Rolling {win}-month Sharpe")
    ax.legend(ncol=3, loc="upper left")
    return fig


def fig_weights(scheds: Dict[str, pd.DataFrame], labels: Dict[str, str]):
    fig, axes = plt.subplots(1, len(scheds), figsize=(3.4 * len(scheds), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, s) in zip(axes, scheds.items()):
        cols = list(s.columns)
        ax.stackplot(s.index, (s[cols] * 100).T.values, colors=PALETTE[:len(cols)],
                     labels=[labels.get(c, c) for c in cols], edgecolor="white", linewidth=0.6,
                     step="post")
        ax.set_title(k); ax.set_ylim(0, 100)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("Target weight (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def fig_tracking(te: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    if "TE2_sd_ann" in te:
        s = te["TE2_sd_ann"].dropna() * 100
        axes[0].barh(s.index, s.values, color=PALETTE[0], height=0.6)
        axes[0].set_xlabel("Tracking error, SD of differences (% p.a.)")
    if "pd_mean_abs" in te:
        s = te["pd_mean_abs"].dropna() * 100
        axes[1].barh(s.index, s.values, color=PALETTE[1], height=0.6)
        axes[1].set_xlabel("Mean |premium/discount| to NAV (%)")
    fig.tight_layout()
    return fig
