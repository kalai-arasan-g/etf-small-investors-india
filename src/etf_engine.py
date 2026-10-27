"""
etf_engine.py
=============
Core engine for the study

    "Do Index ETFs Help Small Investors Diversify? Evidence from India with
     Realistic Costs, Taxes and Whole-Unit Purchases"
    Kalaiarasan G and A. V. Shyam, Amrita Vishwa Vidyapeetham, Coimbatore

Contents
--------
1. Indian transaction-cost model (brokerage, STT, exchange, SEBI, stamp, GST, DP, spread)
2. FIFO tax-lot engine for Indian capital-gains tax (current FY2026-27 rules and a
   historical, date-dependent regime used as a robustness check)
3. Portfolio simulator with whole-unit (integer) purchases, lump-sum and SIP
   contributions, rebalancing, tax payment each July, final liquidation
4. ETF weighting rules (1/N, minimum-variance, constrained max-Sharpe, risk parity)
   with sample, Ledoit-Wolf shrinkage and DCC-GARCH covariance
5. Performance metrics (Sharpe, Sortino, MDD, CVaR, CEQ, turnover, XIRR ...)
6. Statistical tests: Ledoit-Wolf (2008) Sharpe-difference bootstrap, Ledoit-Wolf
   (2011) variance-difference bootstrap, Romano-Wolf (2005) stepdown p-values
7. Tracking error / premium-discount / liquidity measures for ETFs

All monetary amounts are in Indian rupees (INR). Monthly frequency unless stated.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

EQUITY_TYPES = {"stock", "eq_etf"}
NONEQUITY_LISTED = {"gold_etf", "intl_etf"}
ASSET_TYPES = {"stock", "eq_etf", "gold_etf", "debt_etf", "intl_etf"}


# =============================================================================
# 1. Transaction costs
# =============================================================================
@dataclass
class CostModel:
    """Indian delivery-trade cost model. Rates are fractions of trade value.

    brokerage : "zero" (discount broker, free delivery) or "flat" (flat fee per order)
    """
    brokerage: str = "zero"
    flat_fee: float = 20.0
    stt: Dict[str, tuple] = field(default_factory=lambda: {
        "stock": (0.001, 0.001),      # 0.1% buy and sell (delivery)
        "eq_etf": (0.0, 0.00001),     # 0.001% on sale of equity-oriented fund units
        "gold_etf": (0.0, 0.0),
        "debt_etf": (0.0, 0.0),
        "intl_etf": (0.0, 0.0),
    })
    exchange: float = 0.0000297       # NSE cash-market transaction charge
    sebi: float = 0.000001            # Rs 10 per crore
    stamp_buy: float = 0.00015        # 0.015% on delivery purchases
    gst: float = 0.18                 # on brokerage + exchange + SEBI fee (+ DP)
    dp_charge_per_sell: float = 15.93 # depository charge per scrip per sell day (incl. GST)
    half_spread: Dict[str, float] = field(default_factory=lambda: {
        "stock": 0.0010, "eq_etf": 0.0005, "gold_etf": 0.0010,
        "debt_etf": 0.0015, "intl_etf": 0.0020})
    extra_bps: float = 0.0            # robustness: extra proportional cost (bps)
    frictionless: bool = False

    # fixed part of an order (does not scale with value)
    def fixed(self, side: str) -> float:
        if self.frictionless:
            return 0.0
        brk = self.flat_fee if self.brokerage == "flat" else 0.0
        dp = self.dp_charge_per_sell if side == "sell" else 0.0
        return brk * (1 + self.gst) + dp

    # proportional part of an order
    def prop(self, side: str, asset_type: str) -> float:
        if self.frictionless:
            return 0.0
        b, s = self.stt[asset_type]
        stt = b if side == "buy" else s
        stamp = self.stamp_buy if side == "buy" else 0.0
        return (stt + (self.exchange + self.sebi) * (1 + self.gst) + stamp
                + self.half_spread[asset_type] + self.extra_bps / 1e4)

    def breakdown(self, side: str, asset_type: str, value: float) -> Dict[str, float]:
        """Itemised cost of one order of `value` rupees."""
        if self.frictionless or value <= 0:
            return {"total": 0.0, "stt": 0.0}
        b, s = self.stt[asset_type]
        brk = self.flat_fee if self.brokerage == "flat" else 0.0
        out = {
            "brokerage": brk,
            "stt": value * (b if side == "buy" else s),
            "exchange": value * self.exchange,
            "sebi": value * self.sebi,
            "stamp": value * self.stamp_buy if side == "buy" else 0.0,
            "dp": self.dp_charge_per_sell if side == "sell" else 0.0,
            "spread": value * (self.half_spread[asset_type] + self.extra_bps / 1e4),
        }
        out["gst"] = self.gst * (out["brokerage"] + out["exchange"] + out["sebi"])
        out["total"] = sum(out.values())
        return out


# =============================================================================
# 2. Tax engine
# =============================================================================
DATE_LTCG_EQUITY = pd.Timestamp("2018-04-01")   # equity LTCG becomes taxable (10%)
DATE_GRANDFATHER = pd.Timestamp("2018-01-31")   # FMV date for grandfathering
DATE_FA2024 = pd.Timestamp("2024-07-23")        # Finance (No.2) Act 2024 rates
DATE_50AA = pd.Timestamp("2023-04-01")          # specified mutual fund rule
DATE_50AA_REDEF = pd.Timestamp("2025-04-01")    # 50AA narrowed to >65% debt funds

DEFAULT_CII = {2014: 240, 2015: 254, 2016: 264, 2017: 272, 2018: 280, 2019: 289,
               2020: 301, 2021: 317, 2022: 331, 2023: 348, 2024: 363, 2025: 376}


def fiscal_year(d: pd.Timestamp) -> int:
    """Indian financial year labelled by its starting calendar year (FY2025 = Apr25-Mar26)."""
    return d.year if d.month >= 4 else d.year - 1


@dataclass
class TaxEngine:
    """FIFO capital-gains tax for a resident individual.

    regime : "current"    -> FY2026-27 rules applied to every sale
             "historical" -> rules in force on each sale/purchase date
             "none"       -> no tax
    slab   : marginal income-tax slab rate for slab-taxed gains (debt, short-term gold/intl)
    """
    regime: str = "current"
    slab: float = 0.20
    cess: float = 0.04
    eq_st_rate: float = 0.20
    eq_lt_rate: float = 0.125
    eq_lt_exemption: float = 125000.0
    oth_lt_rate: float = 0.125
    cii: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_CII))
    gf_prices: Optional[Dict[str, float]] = None   # FMV on 31-Jan-2018 per asset
    realized: Dict[int, list] = field(default_factory=dict)
    cf_st: float = 0.0     # short-term loss carried forward
    cf_lt: float = 0.0     # long-term loss carried forward
    paid: Dict[int, float] = field(default_factory=dict)

    # --- classification of one realised lot ---------------------------------
    def _classify(self, atype, buy_date, sell_date, sale_value, cost_basis, asset, units=0.0):
        hold = (sell_date - buy_date).days
        if self.regime == "current":
            if atype in EQUITY_TYPES:
                return ("eq_lt", self.eq_lt_rate) if hold > 365 else ("eq_st", self.eq_st_rate), cost_basis
            if atype in NONEQUITY_LISTED:
                return ("oth_lt", self.oth_lt_rate) if hold > 365 else ("slab", self.slab), cost_basis
            return ("slab", self.slab), cost_basis           # debt ETF (Sec 50AA)

        # ---------------- historical regime ----------------
        if atype in EQUITY_TYPES:
            if hold > 365:
                if sell_date < DATE_LTCG_EQUITY:
                    return ("eq_lt_exempt", 0.0), cost_basis
                if buy_date <= DATE_GRANDFATHER and self.gf_prices and asset in self.gf_prices:
                    fmv = self.gf_prices[asset] * units
                    cost_basis = max(cost_basis, min(fmv, sale_value))
                rate = 0.10 if sell_date < DATE_FA2024 else 0.125
                return ("eq_lt", rate), cost_basis
            return ("eq_st", 0.15 if sell_date < DATE_FA2024 else 0.20), cost_basis

        # non-equity: gold, international, debt
        if atype == "debt_etf" and buy_date >= DATE_50AA:
            return ("slab", self.slab), cost_basis
        if atype in NONEQUITY_LISTED and buy_date >= DATE_50AA and sell_date < DATE_50AA_REDEF:
            return ("slab", self.slab), cost_basis
        if sell_date < DATE_FA2024:
            if hold > 1095:   # 36 months, 20% with indexation
                fb, fs = fiscal_year(buy_date), fiscal_year(sell_date)
                if fb in self.cii and fs in self.cii:
                    cost_basis = cost_basis * self.cii[fs] / self.cii[fb]
                return ("oth_lt_idx", 0.20), cost_basis
            return ("slab", self.slab), cost_basis
        if hold > 365:
            return ("oth_lt", 0.125), cost_basis
        return ("slab", self.slab), cost_basis

    def record(self, asset, atype, buy_date, sell_date, sale_value, cost_basis, units=0.0):
        if self.regime == "none":
            return
        (bucket, rate), basis = self._classify(atype, buy_date, sell_date, sale_value, cost_basis, asset, units)
        fy = fiscal_year(sell_date)
        self.realized.setdefault(fy, []).append((bucket, rate, sale_value - basis))

    def exemption(self, fy: int) -> float:
        if self.regime == "current":
            return self.eq_lt_exemption
        return 125000.0 if fy >= 2024 else 100000.0

    def settle(self, fy: int) -> float:
        """Compute tax payable for financial year `fy` (with set-off and carry-forward)."""
        if self.regime == "none" or fy in self.paid:
            return 0.0
        rows = self.realized.get(fy, [])
        net: Dict[tuple, float] = {}
        for b, r, g in rows:
            net[(b, r)] = net.get((b, r), 0.0) + g
        st_b = {"eq_st", "slab"}
        st_loss = self.cf_st + sum(-v for (b, r), v in net.items() if v < 0 and b in st_b)
        lt_loss = self.cf_lt + sum(-v for (b, r), v in net.items() if v < 0 and b not in st_b)
        pos = [[b, r, v] for (b, r), v in net.items() if v > 0 and b != "eq_lt_exempt"]
        pos.sort(key=lambda x: -x[1])
        for p in pos:                                  # LT losses -> LT gains only
            if p[0] not in st_b and lt_loss > 0:
                use = min(lt_loss, p[2]); p[2] -= use; lt_loss -= use
        for p in pos:                                  # ST losses -> any gains
            if st_loss > 0:
                use = min(st_loss, p[2]); p[2] -= use; st_loss -= use
        self.cf_st, self.cf_lt = st_loss, lt_loss
        ex = self.exemption(fy)
        for p in [q for q in pos if q[0] == "eq_lt"]:  # Sec 112A exemption
            use = min(ex, p[2]); p[2] -= use; ex -= use
        tax = sum(v * r for b, r, v in pos) * (1 + self.cess)
        self.paid[fy] = tax
        return tax


# =============================================================================
# 3. Portfolio simulator
# =============================================================================
@dataclass
class Lot:
    units: float
    cost_unit: float      # cost basis per unit (price + allowable costs)
    date: pd.Timestamp


class Portfolio:
    def __init__(self, assets: List[str], types: Dict[str, str], cost: CostModel,
                 tax: TaxEngine, integer: bool = True, min_order: float = 0.0):
        self.assets = list(assets)
        self.types = types
        self.cost = cost
        self.tax = tax
        self.integer = integer
        self.min_order = min_order
        self.units = {a: 0.0 for a in self.assets}
        self.lots: Dict[str, List[Lot]] = {a: [] for a in self.assets}
        self.cash = 0.0
        self.period_cost = 0.0
        self.period_traded = 0.0
        self.period_sold = 0.0
        self.total_cost = 0.0
        self.last_price: Dict[str, float] = {}

    # ---- valuation ---------------------------------------------------------
    def holdings_value(self, px: pd.Series) -> float:
        v = 0.0
        for a, u in self.units.items():
            if u:
                p = px.get(a, np.nan)
                v += u * (p if np.isfinite(p) else self.last_price.get(a, 0.0))
        return v

    def value(self, px: pd.Series) -> float:
        return self.holdings_value(px) + self.cash

    # ---- trading -----------------------------------------------------------
    def sell(self, a: str, units: float, price: float, date: pd.Timestamp):
        units = min(units, self.units[a])
        if units <= 1e-12:
            return
        atype = self.types[a]
        val = units * price
        bd = self.cost.breakdown("sell", atype, val)
        c = bd["total"]
        proceeds = val - c
        allow = c - bd["stt"]                    # STT is not deductible (Sec 48)
        rem = units
        while rem > 1e-12 and self.lots[a]:
            lot = self.lots[a][0]
            take = min(rem, lot.units)
            frac = take / units
            self.tax.record(a, atype, lot.date, date, val * frac - allow * frac, take * lot.cost_unit, take)
            lot.units -= take
            rem -= take
            if lot.units <= 1e-12:
                self.lots[a].pop(0)
        self.units[a] -= units
        if self.units[a] < 1e-9:
            self.units[a] = 0.0
        self.cash += proceeds
        self.period_cost += c
        self.total_cost += c
        self.period_traded += val
        self.period_sold += val

    def buy(self, a: str, units: float, price: float, date: pd.Timestamp):
        if units <= 1e-12:
            return
        atype = self.types[a]
        val = units * price
        bd = self.cost.breakdown("buy", atype, val)
        c = bd["total"]
        self.cash -= val + c
        self.units[a] += units
        self.lots[a].append(Lot(units, (val + c - bd["stt"]) / units, date))
        self.period_cost += c
        self.total_cost += c
        self.period_traded += val

    def liquidate_missing(self, px: pd.Series, date):
        """Sell holdings whose price is missing (delisted) at last known price."""
        for a, u in self.units.items():
            if u and not np.isfinite(px.get(a, np.nan)):
                self.sell(a, u, self.last_price.get(a, 0.0), date)

    def _buy_plan(self, target_val: Dict[str, float], px: pd.Series, cash: float,
                  single: bool = False) -> Dict[str, float]:
        """Whole-unit purchase plan that moves holdings toward target values using `cash`.
        single=True: one order only, the whole amount into the most underweight asset
        (how a small SIP investor avoids paying a flat fee on several orders)."""
        deficits = {}
        for a, tv in target_val.items():
            p = px.get(a, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue
            d = tv - self.units[a] * p
            if d > 0:
                deficits[a] = d
        if not deficits:
            return {}
        if single:
            best, best_gap = None, -1.0
            for a, d in deficits.items():
                unit_cost = px[a] * (1 + self.cost.prop("buy", self.types[a]))
                if unit_cost + self.cost.fixed("buy") <= cash and d / max(target_val[a], 1e-9) > best_gap:
                    best, best_gap = a, d / max(target_val[a], 1e-9)
            if best is None:
                return {}
            unit_cost = px[best] * (1 + self.cost.prop("buy", self.types[best]))
            u = (cash - self.cost.fixed("buy")) / unit_cost
            u = math.floor(u) if self.integer else u
            return {best: u} if u > 0 else {}
        if not self.integer:
            tot = sum(deficits.values())
            plan = {}
            for a, d in deficits.items():
                pr = self.cost.prop("buy", self.types[a])
                spend = cash * d / tot if tot > cash else d
                plan[a] = max(0.0, spend / (px[a] * (1 + pr)))
            return plan

        def need(plan):
            s = 0.0
            for a, u in plan.items():
                if u > 0:
                    v = u * px[a]
                    s += v * (1 + self.cost.prop("buy", self.types[a])) + self.cost.fixed("buy")
            return s

        plan = {a: math.floor(d / px[a]) for a, d in deficits.items()}
        tot = need(plan)
        if tot > cash and tot > 0:
            f = cash / tot
            plan = {a: math.floor(u * f) for a, u in plan.items()}
        while need(plan) > cash and any(u > 0 for u in plan.values()):
            a = max(plan, key=lambda k: plan[k] * px[k])
            plan[a] -= 1
        # greedy top-up: one unit at a time to the largest remaining shortfall
        while True:
            spare = cash - need(plan)
            best, best_gap = None, 0.0
            for a, d in deficits.items():
                gap = d - plan.get(a, 0) * px[a]
                if gap < 0.5 * px[a]:
                    continue
                marg = px[a] * (1 + self.cost.prop("buy", self.types[a]))
                if plan.get(a, 0) == 0:
                    marg += self.cost.fixed("buy")
                if marg <= spare and gap / max(target_val[a], 1e-9) > best_gap:
                    best, best_gap = a, gap / max(target_val[a], 1e-9)
            if best is None:
                break
            plan[best] = plan.get(best, 0) + 1
        if self.min_order > 0:
            plan = {a: u for a, u in plan.items() if u * px[a] >= self.min_order}
        return {a: u for a, u in plan.items() if u > 0}

    def trade_to(self, w: pd.Series, px: pd.Series, date, full: bool = True, single: bool = False):
        """Move toward target weights `w`. full=True sells overweights; False uses cash only."""
        w = w[w.index.isin(self.assets)]
        valid = [a for a in w.index if np.isfinite(px.get(a, np.nan)) and px[a] > 0]
        w = w.loc[valid]
        if w.sum() <= 0:
            return
        w = w / w.sum()
        W = self.value(px)
        target_val = {a: float(w.get(a, 0.0)) * W for a in self.assets}
        if full:
            for a in self.assets:
                u = self.units[a]
                if not u:
                    continue
                p = px.get(a, np.nan)
                if not np.isfinite(p):
                    continue
                tgt_units = target_val.get(a, 0.0) / p
                if self.integer:
                    tgt_units = math.floor(tgt_units)
                excess = u - tgt_units
                # avoid tiny sells: only sell if excess value > 0.5% of wealth or target is zero
                if excess > 1e-9 and (excess * p > 0.005 * W or target_val.get(a, 0.0) == 0):
                    self.sell(a, excess, p, date)
        plan = self._buy_plan({a: target_val[a] for a in w.index}, px, max(self.cash, 0.0), single=single)
        for a, u in plan.items():
            self.buy(a, u, px[a], date)

    def raise_cash(self, amount: float, px: pd.Series, date):
        """Sell pro-rata to raise `amount` rupees (used to pay tax)."""
        hv = self.holdings_value(px)
        if hv <= 0:
            return
        frac = min(1.0, amount * 1.02 / hv)
        for a in self.assets:
            u = self.units[a]
            p = px.get(a, np.nan)
            if u and np.isfinite(p):
                s = u * frac
                s = math.ceil(s) if self.integer else s
                self.sell(a, min(s, u), p, date)


@dataclass
class SimResult:
    values: pd.Series          # end-of-month portfolio value (after trades and tax)
    contrib: pd.Series         # contribution invested at each date
    returns: pd.Series         # time-weighted monthly returns
    costs: pd.Series
    taxes: pd.Series
    traded: pd.Series
    sold: pd.Series            # value sold (rebalancing / tax / delisting sales)
    weights: pd.DataFrame      # end-of-month weights
    total_contrib: float
    terminal_pre_tax: float
    terminal_post_tax: float
    total_cost: float
    total_tax: float
    final_liq_tax: float
    xirr: float


def simulate(prices: pd.DataFrame, types: Dict[str, str], weight_fn: Callable,
             rebalance_dates: set, contributions: pd.Series, cost: CostModel,
             tax: TaxEngine, integer: bool = True, threshold: Optional[float] = None,
             tax_month: int = 7, liquidate_end: bool = True, sip_single_order: bool = False) -> SimResult:
    """Run one portfolio through time.

    prices         : month-end tradable prices (index = dates, columns = assets)
    weight_fn(t)   : returns target weights (pd.Series) at date index position t
    rebalance_dates: dates on which a full rebalance happens (the first date always trades)
    contributions  : rupees added at each date (lump sum -> only first date)
    threshold      : if given, also rebalance when any weight drifts > threshold
    sip_single_order: in contribution-only months place one order (most underweight asset)
    """
    dates = prices.index
    assets = list(prices.columns)
    pf = Portfolio(assets, types, cost, tax, integer=integer)
    vals, rets, costs_, taxes_, trd, sold, wts = [], [], [], [], [], [], []
    contrib = contributions.reindex(dates).fillna(0.0)
    target = None
    prev_v = None
    for t, d in enumerate(dates):
        px = prices.iloc[t]
        pf.period_cost = 0.0
        pf.period_traded = 0.0
        pf.period_sold = 0.0
        tax_paid = 0.0
        pf.liquidate_missing(px, d)
        for a in assets:
            if np.isfinite(px[a]):
                pf.last_price[a] = px[a]
        # pay last financial year's tax in July
        if d.month == tax_month:
            due = tax.settle(fiscal_year(d) - 1)
            if due > 0:
                if pf.cash < due:
                    pf.raise_cash(due - pf.cash, px, d)
                pf.cash -= due
                tax_paid += due
        c = float(contrib.iloc[t])
        pf.cash += c
        do_full = (t == 0) or (d in rebalance_dates)
        if do_full or target is None:
            target = weight_fn(t)
        if threshold is not None and not do_full and target is not None:
            W = pf.value(px)
            if W > 0:
                cur = pd.Series({a: pf.units[a] * px[a] / W if np.isfinite(px[a]) else 0 for a in assets})
                if (cur.reindex(target.index).fillna(0) - target).abs().max() > threshold:
                    do_full = True
        if do_full:
            pf.trade_to(target, px, d, full=True)
        elif c > 0:
            pf.trade_to(target, px, d, full=False, single=sip_single_order)
        v = pf.value(px)
        if prev_v is not None and prev_v > 0:
            rets.append((v - c) / prev_v - 1.0)
        else:
            rets.append(np.nan)
        prev_v = v
        vals.append(v)
        costs_.append(pf.period_cost)
        taxes_.append(tax_paid)
        trd.append(pf.period_traded)
        sold.append(pf.period_sold)
        wts.append({a: (pf.units[a] * px[a] / v if v > 0 and np.isfinite(px[a]) else 0.0) for a in assets})

    values = pd.Series(vals, index=dates)
    terminal_pre = values.iloc[-1]
    final_tax = 0.0
    if liquidate_end:
        d = dates[-1]
        px = prices.iloc[-1]
        for a in assets:
            if pf.units[a]:
                p = px[a] if np.isfinite(px[a]) else pf.last_price.get(a, 0.0)
                pf.sell(a, pf.units[a], p, d)
        for fy in sorted(set(tax.realized) | {fiscal_year(d)}):
            if fy not in tax.paid:
                final_tax += tax.settle(fy)
        terminal_post = pf.cash - final_tax
    else:
        terminal_post = terminal_pre
    total_tax = float(np.nansum(taxes_)) + final_tax
    cfs = -contrib.values.copy()
    cfs[-1] += terminal_post
    return SimResult(
        values=values, contrib=contrib, returns=pd.Series(rets, index=dates).iloc[1:],
        costs=pd.Series(costs_, index=dates), taxes=pd.Series(taxes_, index=dates),
        traded=pd.Series(trd, index=dates), sold=pd.Series(sold, index=dates), weights=pd.DataFrame(wts, index=dates),
        total_contrib=float(contrib.sum()), terminal_pre_tax=float(terminal_pre),
        terminal_post_tax=float(terminal_post), total_cost=float(pf.total_cost),
        total_tax=total_tax, final_liq_tax=final_tax, xirr=xirr_monthly(cfs))


def rebalance_schedule(dates: pd.DatetimeIndex, freq: str) -> set:
    """freq: 'M', 'Q', 'A' (anniversary of first date) or 'none'."""
    if freq == "none":
        return set()
    if freq == "M":
        return set(dates)
    step = {"Q": 3, "A": 12}[freq]
    return set(dates[::step])


def xirr_monthly(cfs: np.ndarray) -> float:
    """Annualised money-weighted return from equally spaced monthly cash flows."""
    t = np.arange(len(cfs))
    def npv(r):
        return np.sum(cfs / (1 + r) ** (t / 12.0))
    try:
        return brentq(npv, -0.99, 10.0)
    except Exception:
        return np.nan


# =============================================================================
# 4. Weighting rules and covariance estimators
# =============================================================================
def cov_sample(R: pd.DataFrame) -> np.ndarray:
    return np.cov(R.values, rowvar=False)


def cov_ledoit_wolf(R: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(R.values).covariance_


def _garch11_fit(x: np.ndarray):
    from arch import arch_model
    am = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    std_resid = res.std_resid
    fvar = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    return np.asarray(std_resid), float(fvar)


def dcc_forecast(Rd: pd.DataFrame, horizon_days: int = 21) -> np.ndarray:
    """One-step-ahead DCC(1,1)-GARCH(1,1) covariance from daily returns, scaled to a month.

    Engle (2002) two-step estimator: univariate GARCH(1,1), then DCC parameters by
    quasi-maximum likelihood on standardised residuals.
    """
    X = Rd.dropna().values * 100.0
    n, k = X.shape
    Z = np.empty_like(X)
    fv = np.empty(k)
    for i in range(k):
        Z[:, i], fv[i] = _garch11_fit(X[:, i])
    Z = np.nan_to_num(Z)
    S = np.corrcoef(Z, rowvar=False)

    def qlik(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        Q = S.copy()
        ll = 0.0
        for t in range(1, n):
            z = Z[t - 1][:, None]
            Q = (1 - a - b) * S + a * (z @ z.T) + b * Q
            d = np.sqrt(np.diag(Q))
            R = Q / np.outer(d, d)
            try:
                sign, logdet = np.linalg.slogdet(R)
                ll += logdet + Z[t] @ np.linalg.solve(R, Z[t])
            except np.linalg.LinAlgError:
                return 1e10
        return 0.5 * ll

    res = minimize(qlik, x0=[0.03, 0.94], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 300})
    a, b = res.x if res.fun < 1e9 else (0.0, 0.0)
    Q = S.copy()
    for t in range(1, n + 1):
        z = Z[t - 1][:, None]
        Q = (1 - a - b) * S + a * (z @ z.T) + b * Q
    d = np.sqrt(np.diag(Q))
    R = Q / np.outer(d, d)
    sd = np.sqrt(fv) / 100.0
    H = R * np.outer(sd, sd)
    return H * horizon_days


def w_equal(assets) -> pd.Series:
    return pd.Series(1.0 / len(assets), index=assets)


def _solve(obj, n, cap):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bnds = [(0.0, cap)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, 0, None)
    return w / w.sum()


def w_minvar(cov: np.ndarray, assets, cap: float = 1.0) -> pd.Series:
    n = len(assets)
    cap = max(cap, 1.0 / n)
    return pd.Series(_solve(lambda w: w @ cov @ w, n, cap), index=assets)


def w_maxsharpe(mu: np.ndarray, cov: np.ndarray, rf: float, assets, cap: float = 1.0) -> pd.Series:
    n = len(assets)
    cap = max(cap, 1.0 / n)
    ex = mu - rf
    if np.all(ex <= 0):
        return w_minvar(cov, assets, cap)
    return pd.Series(_solve(lambda w: -(w @ ex) / np.sqrt(w @ cov @ w + 1e-16), n, cap), index=assets)


def w_riskparity(cov: np.ndarray, assets) -> pd.Series:
    n = len(assets)
    def obj(w):
        pv = w @ cov @ w
        rc = w * (cov @ w)
        return np.sum((rc - pv / n) ** 2) * 1e4
    return pd.Series(_solve(obj, n, 1.0), index=assets)


def diversification_ratio(w: pd.Series, cov: np.ndarray) -> float:
    w = w.values
    sd = np.sqrt(np.diag(cov))
    return float((w @ sd) / np.sqrt(w @ cov @ w))


# =============================================================================
# 5. Performance metrics
# =============================================================================
def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min())


def perf_metrics(r: pd.Series, rf: pd.Series, gamma: float = 1.0) -> Dict[str, float]:
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0.0)
    ex = r - rf
    n = len(r)
    if n < 3:
        return {}
    ann_ret = float((1 + r).prod() ** (12.0 / n) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(12))
    sd_ex = ex.std(ddof=1)
    sharpe = float(ex.mean() / sd_ex * np.sqrt(12)) if sd_ex > 0 else np.nan
    dd = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    sortino = float(ex.mean() / dd * np.sqrt(12)) if dd > 0 else np.nan
    q = np.quantile(r, 0.05)
    cvar = float(-r[r <= q].mean())
    ceq = float((ex.mean() - gamma / 2 * ex.var(ddof=1)) * 12)
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_drawdown(r), "cvar95_monthly": cvar, "ceq": ceq,
            "skew": float(r.skew()), "kurtosis": float(r.kurt()), "n_months": n}


def sim_summary(res: SimResult, rf: pd.Series) -> Dict[str, float]:
    m = perf_metrics(res.returns, rf)
    avg_v = res.values.replace(0, np.nan).mean()
    m.update({
        "total_contrib": res.total_contrib,
        "terminal_pre_tax": res.terminal_pre_tax,
        "terminal_post_tax": res.terminal_post_tax,
        "wealth_multiple_post_tax": res.terminal_post_tax / res.total_contrib if res.total_contrib else np.nan,
        "xirr_post_tax": res.xirr,
        "total_cost": res.total_cost,
        "total_tax": res.total_tax,
        "cost_pct_contrib": res.total_cost / res.total_contrib if res.total_contrib else np.nan,
        "tax_pct_contrib": res.total_tax / res.total_contrib if res.total_contrib else np.nan,
        # tax as a share of the gross gain (post-tax wealth + all tax paid - contributions)
        "tax_pct_gain": (res.total_tax / (res.terminal_post_tax + res.total_tax - res.total_contrib)
                         if (res.terminal_post_tax + res.total_tax - res.total_contrib) > 0 else np.nan),
        "annual_turnover": float(res.sold.sum() / avg_v / (len(res.values) / 12)) if avg_v else np.nan,
        "avg_cash_weight": float(1 - res.weights.sum(axis=1).mean()),
    })
    return m


# =============================================================================
# 6. Statistical tests
# =============================================================================
def _qs_kernel(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    z = 6 * np.pi * x[nz] / 5
    out[nz] = 25 / (12 * np.pi ** 2 * x[nz] ** 2) * (np.sin(z) / z - np.cos(z))
    return out


def hac_qs(Y: np.ndarray) -> np.ndarray:
    """HAC covariance (Quadratic-Spectral kernel, Andrews 1991 AR(1) bandwidth)."""
    T, k = Y.shape
    Y = Y - Y.mean(axis=0)
    num = den = 0.0
    for i in range(k):
        y = Y[:, i]
        if np.var(y) == 0:
            continue
        rho = np.clip(np.dot(y[1:], y[:-1]) / np.dot(y[:-1], y[:-1]), -0.97, 0.97)
        s2 = np.var(y[1:] - rho * y[:-1])
        num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        den += s2 ** 2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    bw = 1.3221 * (alpha2 * T) ** 0.2 if alpha2 > 0 else 1.0
    Psi = Y.T @ Y / T
    for j in range(1, T):
        kw = _qs_kernel(j / bw)
        if abs(kw) < 1e-8 and j > 5 * bw:
            break
        G = Y[j:].T @ Y[:-j] / T
        Psi += kw * (G + G.T)
    return Psi * T / (T - 4)


def _sr_stat(r1, r2):
    m1, m2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    return m1 / np.sqrt(g1 - m1 ** 2) - m2 / np.sqrt(g2 - m2 ** 2), (m1, m2, g1, g2)


def _sr_grad(m1, m2, g1, g2):
    return np.array([g1 / (g1 - m1 ** 2) ** 1.5, -g2 / (g2 - m2 ** 2) ** 1.5,
                     -0.5 * m1 / (g1 - m1 ** 2) ** 1.5, 0.5 * m2 / (g2 - m2 ** 2) ** 1.5])


def _var_stat(r1, r2):
    m1, m2 = r1.mean(), r2.mean()
    g1, g2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    return np.log(g1 - m1 ** 2) - np.log(g2 - m2 ** 2), (m1, m2, g1, g2)


def _var_grad(m1, m2, g1, g2):
    return np.array([-2 * m1 / (g1 - m1 ** 2), 2 * m2 / (g2 - m2 ** 2),
                     1 / (g1 - m1 ** 2), -1 / (g2 - m2 ** 2)])


def _cbb_indices(T, b, M, rng):
    l = int(np.ceil(T / b))
    starts = rng.integers(0, T, size=(M, l))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % T
    return idx.reshape(M, l * b), l


def _boot_test(r1, r2, stat_fn, grad_fn, b, M, seed):
    r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
    T = len(r1)
    d_hat, (m1, m2, g1, g2) = stat_fn(r1, r2)
    Y = np.column_stack([r1 - m1, r2 - m2, r1 ** 2 - g1, r2 ** 2 - g2])
    grad = grad_fn(m1, m2, g1, g2)
    se = np.sqrt(grad @ hac_qs(Y) @ grad / T)
    rng = np.random.default_rng(seed)
    idx, l = _cbb_indices(T, b, M, rng)
    a1, a2 = r1[idx], r2[idx]
    Tb = idx.shape[1]
    mm1, mm2 = a1.mean(1), a2.mean(1)
    gg1, gg2 = (a1 ** 2).mean(1), (a2 ** 2).mean(1)
    if stat_fn is _sr_stat:
        d_star = mm1 / np.sqrt(gg1 - mm1 ** 2) - mm2 / np.sqrt(gg2 - mm2 ** 2)
        G = np.column_stack([gg1 / (gg1 - mm1 ** 2) ** 1.5, -gg2 / (gg2 - mm2 ** 2) ** 1.5,
                             -0.5 * mm1 / (gg1 - mm1 ** 2) ** 1.5, 0.5 * mm2 / (gg2 - mm2 ** 2) ** 1.5])
    else:
        d_star = np.log(gg1 - mm1 ** 2) - np.log(gg2 - mm2 ** 2)
        G = np.column_stack([-2 * mm1 / (gg1 - mm1 ** 2), 2 * mm2 / (gg2 - mm2 ** 2),
                             1 / (gg1 - mm1 ** 2), -1 / (gg2 - mm2 ** 2)])
    Ys = np.stack([a1 - mm1[:, None], a2 - mm2[:, None],
                   a1 ** 2 - gg1[:, None], a2 ** 2 - gg2[:, None]], axis=2)  # M x Tb x 4
    blocks = Ys.reshape(M, l, b, 4).sum(axis=2) / np.sqrt(b)                  # M x l x 4
    Psi = np.einsum("mli,mlj->mij", blocks, blocks) / l
    se_star = np.sqrt(np.einsum("mi,mij,mj->m", G, Psi, G) / Tb)
    t_hat = abs(d_hat) / se
    t_star = np.abs(d_star - d_hat) / se_star
    p = (np.sum(t_star >= t_hat) + 1) / (M + 1)
    return {"diff": float(d_hat), "se": float(se), "t": float(d_hat / se), "p_value": float(p),
            "_dstar": d_star, "_sestar": se_star}


def ledoit_wolf_sharpe_test(r1, r2, b: int = 5, M: int = 4999, seed: int = 42) -> Dict[str, float]:
    """H0: equal Sharpe ratios (Ledoit & Wolf 2008, studentised circular block bootstrap).
    Input: excess returns. `diff` is the per-period (monthly) Sharpe difference."""
    out = _boot_test(r1, r2, _sr_stat, _sr_grad, b, M, seed)
    out["diff_annualised"] = out["diff"] * np.sqrt(12)
    return {k: v for k, v in out.items() if not k.startswith("_")}


def ledoit_wolf_variance_test(r1, r2, b: int = 5, M: int = 4999, seed: int = 42) -> Dict[str, float]:
    """H0: equal variances (Ledoit & Wolf 2011). `diff` = log(var1) - log(var2)."""
    out = _boot_test(r1, r2, _var_stat, _var_grad, b, M, seed)
    return {k: v for k, v in out.items() if not k.startswith("_")}


def romano_wolf_sharpe(bench: np.ndarray, others: Dict[str, np.ndarray], b: int = 5,
                       M: int = 4999, seed: int = 42) -> pd.DataFrame:
    """Romano-Wolf (2005) stepdown adjusted p-values for H0_j: SR_j = SR_bench (two-sided)."""
    names = list(others)
    t_obs, t_boot = [], []
    for nme in names:
        o = _boot_test(others[nme], bench, _sr_stat, _sr_grad, b, M, seed)  # same seed -> same blocks
        t_obs.append(abs(o["t"]))
        t_boot.append(np.abs(o["_dstar"] - o["diff"]) / o["_sestar"])
    t_obs = np.array(t_obs)
    t_boot = np.array(t_boot)          # k x M
    order = np.argsort(-t_obs)
    adj = np.empty(len(names))
    prev = 0.0
    for step, j in enumerate(order):
        active = order[step:]
        maxstat = t_boot[active].max(axis=0)
        p = (np.sum(maxstat >= t_obs[j]) + 1) / (M + 1)
        prev = max(prev, p)
        adj[j] = prev
    return pd.DataFrame({"strategy": names, "abs_t": t_obs, "rw_adj_p": adj})


# =============================================================================
# 7. ETF quality measures
# =============================================================================
def tracking_measures(r_etf: pd.Series, r_idx: pd.Series, periods: int = 252,
                      error_cut: float = 0.20) -> Dict[str, float]:
    """Three tracking-error definitions used in the Indian ETF literature.
    TE1 mean absolute difference, TE2 std of differences, TE3 regression std error.
    Days where ETF and index returns differ by more than `error_cut` (20%) are treated as
    data errors (bad ticks, unadjusted splits) and excluded; their number is reported."""
    df = pd.concat([r_etf, r_idx], axis=1).dropna()
    bad = (df.iloc[:, 0] - df.iloc[:, 1]).abs() > error_cut
    n_bad = int(bad.sum())
    df = df[~bad]
    if len(df) < 30:
        return {}
    e, i = df.iloc[:, 0].values, df.iloc[:, 1].values
    d = e - i
    X = np.column_stack([np.ones_like(i), i])
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    resid = e - X @ beta
    te3 = np.sqrt(resid @ resid / (len(e) - 2))
    yrs = len(df) / periods
    return {
        "TE1_mad_ann": float(np.mean(np.abs(d)) * np.sqrt(periods)),
        "TE2_sd_ann": float(np.std(d, ddof=1) * np.sqrt(periods)),
        "TE3_reg_ann": float(te3 * np.sqrt(periods)),
        "tracking_diff_ann": float(np.prod(1 + e) ** (1 / yrs) - np.prod(1 + i) ** (1 / yrs)),
        "alpha_ann": float(beta[0] * periods), "beta": float(beta[1]),
        "n_obs": int(len(df)), "days_excluded_data_errors": n_bad,
    }


def premium_discount(price: pd.Series, nav: pd.Series) -> Dict[str, float]:
    df = pd.concat([price, nav], axis=1).dropna()
    if len(df) < 30:
        return {}
    pdisc = df.iloc[:, 0] / df.iloc[:, 1] - 1
    bad = pdisc.abs() > 0.5            # |premium| above 50% = NAV/price data error, not a real premium
    n_bad = int(bad.sum())
    pdisc = pdisc[~bad]
    return {"pd_days_excluded_data_errors": n_bad,
            "pd_mean": float(pdisc.mean()), "pd_sd": float(pdisc.std()),
            "pd_mean_abs": float(pdisc.abs().mean()),
            "pct_days_abs_gt_1pct": float((pdisc.abs() > 0.01).mean()),
            "pd_max": float(pdisc.max()), "pd_min": float(pdisc.min())}


def liquidity_measures(close: pd.Series, volume: pd.Series) -> Dict[str, float]:
    df = pd.concat([close, volume], axis=1).dropna()
    df.columns = ["p", "v"]
    tv = df["p"] * df["v"]
    r = df["p"].pct_change().abs()
    amihud = (r / (tv / 1e7)).replace([np.inf, -np.inf], np.nan)
    return {"avg_daily_traded_value_cr": float(tv.mean() / 1e7),
            "median_daily_traded_value_cr": float(tv.median() / 1e7),
            "amihud_illiq": float(amihud.mean()),
            "pct_zero_volume_days": float((df["v"] == 0).mean())}


def dcc_correlations(Rd: pd.DataFrame) -> pd.DataFrame:
    """Full-sample DCC(1,1)-GARCH(1,1) conditional correlations, one column per asset pair."""
    X = Rd.dropna()
    cols = list(X.columns)
    Xv = X.values * 100.0
    n, k = Xv.shape
    Z = np.empty_like(Xv)
    for i in range(k):
        Z[:, i], _ = _garch11_fit(Xv[:, i])
    Z = np.nan_to_num(Z)
    S = np.corrcoef(Z, rowvar=False)

    def path(a, b):
        Q = S.copy()
        out = np.empty((n, k, k))
        out[0] = S
        for t in range(1, n):
            z = Z[t - 1][:, None]
            Q = (1 - a - b) * S + a * (z @ z.T) + b * Q
            d = np.sqrt(np.diag(Q))
            out[t] = Q / np.outer(d, d)
        return out

    def qlik(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        Rs = path(a, b)
        ll = 0.0
        for t in range(1, n):
            sign, logdet = np.linalg.slogdet(Rs[t])
            ll += logdet + Z[t] @ np.linalg.solve(Rs[t], Z[t])
        return 0.5 * ll

    res = minimize(qlik, x0=[0.03, 0.94], method="Nelder-Mead", options={"maxiter": 200})
    a, b = res.x if res.fun < 1e9 else (0.0, 0.0)
    Rs = path(a, b)
    data = {f"{cols[i]} / {cols[j]}": Rs[:, i, j] for i in range(k) for j in range(i + 1, k)}
    df = pd.DataFrame(data, index=X.index)
    df.attrs["dcc_a"], df.attrs["dcc_b"] = float(a), float(b)
    return df
