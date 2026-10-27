"""
make_latex_tables.py - build the manuscript's LaTeX tables from a results run folder.

Usage:  python scripts/make_latex_tables.py results/run_20261026_0831/tables  latex_tables/
Re-run after any new full run; every number in the tables comes from the CSVs.
"""
import os
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/run_20261026_0831/tables"
OUT = sys.argv[2] if len(sys.argv) > 2 else "latex_tables"
os.makedirs(OUT, exist_ok=True)

ETF_ORDER = ["Nifty 50", "Nifty Next 50", "Gold", "G-Sec", "Nasdaq 100"]
STRAT_ORDER = ["Nifty 50 ETF only", "Equity ETFs 1/N", "1/N ETF", "Min-variance ETF", "Max-Sharpe ETF", "Risk-parity ETF"]
STRAT_TEX = {"Nifty 50 ETF only": "Nifty 50 ETF only", "Equity ETFs 1/N": "Equity ETFs, $1/N$", "1/N ETF": "Multi-asset $1/N$",
             "Min-variance ETF": "Minimum variance", "Max-Sharpe ETF": "Maximum Sharpe", "Risk-parity ETF": "Risk parity"}


def rd(name):
    return pd.read_csv(os.path.join(SRC, f"{name}.csv"))


def _neg(t):
    return t.replace("-", "$-$", 1) if t.startswith("-") else t


def pct(x, d=1):
    return "--" if pd.isna(x) else _neg(f"{x * 100:.{d}f}")


def num(x, d=2):
    return "--" if pd.isna(x) else _neg(f"{x:.{d}f}")


def esc(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("Rs ", r"Rs~")


def write(name, caption, label, colspec, header, rows, note, size="\\footnotesize", sideways=False, panels=None):
    env = "sidewaystable" if sideways else "table"
    # sn-jnl breaks when a tabular is boxed inside a table float, so the tabular is typeset into a
    # save box first and scaled down inside the float only if wider than the text (\fittable, main.tex).
    # The one-cell tabular wrapper is needed because sn-jnl writes a table's \label only when the
    # float contains a tabular.
    lines = ["\\sbox{\\tabbox}{%", f"{size}", f"\\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", "\\toprule"]
    lines += header
    lines.append("\\midrule")
    if panels:
        for i, (title, prow) in enumerate(panels):
            if i:
                lines.append("\\midrule")
            ncol = sum(ch in "lrc" for ch in colspec)
            lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
            lines += [" & ".join(r) + " \\\\" for r in prow]
    else:
        lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\botrule", "\\end{tabular}}%", f"\\begin{{{env}}}[!htbp]", f"\\caption{{{caption}}}\\label{{{label}}}",
              "\\centering\\begin{tabular}{@{}c@{}}\\fittable{\\tabbox}\\end{tabular}", f"\\footnotetext{{{note}}}", f"\\end{{{env}}}"]
    with open(os.path.join(OUT, f"{name}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", name)


# ---------------------------------------------------------------- Table 1: sample
t1 = rd("T01_etf_selection").set_index("label")
t2 = rd("T02_descriptive_stats").set_index("ETF")
cls = {"eq_etf": "Equity", "gold_etf": "Gold", "debt_etf": "Govt. bond", "intl_etf": "Intl. equity"}
rows = []
for e in ETF_ORDER:
    a, b = t1.loc[e], t2.loc[e]
    rows.append([e, esc(a["ticker"].replace(".NS", "")), cls[a["type"]], str(a["first_date"])[:7],
                 num(a["median_daily_traded_value_cr_2y"], 2), num(b["ann_return_%"], 1), num(b["ann_vol_%"], 1),
                 num(b["skewness"], 2), num(b["excess_kurtosis"], 2), num(b["jb_p"], 3)])
write("tab_sample", "ETF sample and descriptive statistics of monthly returns", "tab:sample", "llllrrrrrr",
      ["ETF & Ticker & Asset class & Listed from & Traded value & Ann. return & Ann. vol. & Skew. & Ex. kurt. & JB $p$ \\\\",
       " & & & & (Rs crore/day) & (\\%) & (\\%) & & & \\\\"], rows,
      "One ETF per asset class, chosen as the candidate with the longest history among those passing a liquidity screen. "
      "Traded value is the median daily traded value over the last two years. Return statistics use month-end prices, "
      f"{t2['months'].iloc[0]} months from February 2015 to September 2026; G-Sec returns use the ETF's NAV because its market "
      "price is stale (Table~\\ref{tab:tracking}). JB $p$ is the Jarque--Bera normality test $p$-value.")

# ---------------------------------------------------------------- Table 2: tracking and liquidity
t5 = rd("T05_tracking_liquidity").set_index("ETF")
rows = []
for e in ETF_ORDER:
    r = t5.loc[e]
    rows.append([e, pct(r.get("nav_TE2_sd_ann"), 2), pct(r.get("nav_tracking_diff_ann"), 2), num(r.get("nav_beta"), 3),
                 pct(r.get("TE2_sd_ann"), 2), pct(r["pd_mean"], 2), pct(r["pd_mean_abs"], 2), pct(r["pct_days_abs_gt_1pct"], 1),
                 num(r["median_daily_traded_value_cr"], 2), pct(r["pct_zero_volume_days"], 1)])
write("tab_tracking", "Tracking efficiency, premium/discount and liquidity of the selected ETFs", "tab:tracking", "lrrrrrrrrr",
      ["& \\multicolumn{3}{c}{NAV vs. TRI} & Price vs. TRI & \\multicolumn{3}{c}{Price vs. NAV} & \\multicolumn{2}{c}{Liquidity}\\\\",
       "\\cmidrule{2-4}\\cmidrule{5-5}\\cmidrule{6-8}\\cmidrule{9-10}",
       "ETF & TE (\\%) & TD (\\%) & $\\beta$ & TE (\\%) & Mean (\\%) & $|$Mean$|$ (\\%) & Days $>$1\\% & Traded value & Zero-vol. \\\\",
       " & & & & & & & (\\%) & (Rs cr/day) & days (\\%)\\\\"], rows,
      "TE is the annualised standard deviation of daily return differences; TD is the annualised tracking difference "
      "(ETF minus index) and $\\beta$ the slope of ETF on index daily returns, all against the NIFTY Total Return Index (TRI). "
      "No matching NIFTY TRI exists for the gold, G-Sec and Nasdaq ETFs. Days on which ETF and index returns differ by "
      "more than 20\\% (an unadjusted 1:10 unit split in December 2019) are excluded as data errors. "
      "Premium/discount is price over NAV minus one; liquidity is computed from daily NSE prices and volumes, 2015--2026.")

# ---------------------------------------------------------------- Table 3: ETF strategy performance
t7 = rd("T07_etf_performance_main")
cols = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar95_monthly", "annual_turnover", "cost_pct_contrib", "tax_pct_contrib", "xirr_post_tax"]
panels = []
for inv, title in (("SIP Rs 5,000/month", "Panel A: SIP investor, Rs 5,000 per month"), ("Lump Rs 50,000", "Panel B: lump-sum investor, Rs 50,000")):
    d = t7[(t7["investor"] == inv)].set_index(["mode", "strategy"])
    prow = []
    for s in STRAT_ORDER:
        n = d.loc[("costs_tax", s)]
        fr = d.loc[("frictionless", s)]
        prow.append([STRAT_TEX[s], pct(n["ann_return"]), pct(n["ann_vol"]), num(n["sharpe"]), num(fr["sharpe"]), num(n["sortino"]),
                     pct(n["max_drawdown"]), pct(n["cvar95_monthly"]), pct(n["annual_turnover"], 0), pct(n["cost_pct_contrib"], 2),
                     pct(n["tax_pct_contrib"]), pct(n["xirr_post_tax"])])
    panels.append((title, prow))
write("tab_etf_perf", "Out-of-sample performance of ETF strategies, net of transaction costs and taxes", "tab:etfperf", "lrrrrrrrrrrr",
      ["Strategy & Return & Vol. & Sharpe & Sharpe & Sortino & MDD & CVaR & Turn- & Costs & Tax & XIRR \\\\",
       " & (\\%) & (\\%) & (net) & (gross) & & (\\%) & (\\%) & over (\\%) & (\\%) & (\\%) & (\\%)\\\\"], None,
      "January 2018 to September 2026 (105 months); weights estimated on a rolling 36-month window, annual rebalancing, "
      "zero-brokerage account, 20\\% income-tax slab, FY2026--27 capital-gains rules. Return, volatility, Sharpe and Sortino "
      "ratios are annualised from time-weighted monthly returns net of costs and taxes paid; Sharpe (gross) ignores all frictions. "
      "MDD is maximum drawdown; CVaR is the mean of the worst 5\\% of monthly returns. Costs and tax are totals as a "
      "percentage of contributions; XIRR is the money-weighted return on post-tax terminal wealth after liquidation.",
      panels=panels, sideways=True)

# ---------------------------------------------------------------- Table 4: tests among ETF strategies
t9 = rd("T09_tests_etf")
rows = []
for bench in ("1/N ETF", "Nifty 50 ETF only"):
    for s in STRAT_ORDER:
        r = t9[(t9["benchmark"] == bench) & (t9["strategy"] == s)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([STRAT_TEX[bench], STRAT_TEX[s], num(r["Sharpe diff (ann.)"]), num(r["Sharpe p (LW2008)"], 3),
                     num(r["Romano-Wolf adj. p (vs 1/N)"], 3), num(r["log variance ratio"]), num(r["variance p (LW2011)"], 3)])
write("tab_tests", "Tests of differences in Sharpe ratio and variance between ETF strategies", "tab:tests", "llrrrrr",
      ["Benchmark & Strategy & $\\Delta$Sharpe & $p$ (LW) & $p$ (RW) & $\\ln(\\sigma^2_s/\\sigma^2_b)$ & $p$ (LW) \\\\"], rows,
      "$\\Delta$Sharpe is the annualised Sharpe ratio of the strategy minus that of the benchmark, net of costs and taxes "
      "(SIP investor). $p$ (LW): studentised circular block bootstrap of \\citet{ledoitwolf2008} for Sharpe ratios and of "
      "\\citet{ledoitwolf2011} for variances (block length 5, 4,999 replications). $p$ (RW): \\citet{romanowolf2005} stepdown "
      "$p$-values controlling the family-wise error rate across all comparisons with the multi-asset $1/N$ portfolio.")

# ---------------------------------------------------------------- Table 5: random stock portfolios vs ETFs
t8 = rd("T08_stocks_vs_etf")
t10 = rd("T10_tests_etf_vs_stocks")
d = t8[t8["mode"] == "costs_tax"].set_index("N stocks")
rows = []
for N in d.index:
    r = d.loc[N]
    a = t10[(t10["ETF benchmark"] == "Nifty 50 ETF only") & (t10["N stocks"] == N)].iloc[0]
    b = t10[(t10["ETF benchmark"] == "1/N ETF") & (t10["N stocks"] == N)].iloc[0]
    rows.append([str(N), num(r["Sharpe p5"]), num(r["Sharpe median"]), num(r["Sharpe p95"]), pct(r["vol median"]), pct(r["MDD median"]),
                 f"{r['% draws below Nifty 50 ETF Sharpe']:.1f}", f"{a['% not significant']:.1f}",
                 f"{r['% draws below 1/N ETF Sharpe']:.1f}", f"{b['% ETF significantly better (p<0.05)']:.1f}"])
e = d.iloc[0]
write("tab_stocks", "Random direct-equity portfolios versus ETF portfolios", "tab:stocks", "rrrrrrrrrr",
      ["& \\multicolumn{3}{c}{Sharpe ratio} & Vol. & MDD & \\multicolumn{2}{c}{vs. Nifty 50 ETF} & \\multicolumn{2}{c}{vs. multi-asset $1/N$}\\\\",
       "\\cmidrule{2-4}\\cmidrule{7-8}\\cmidrule{9-10}",
       "$N$ & P5 & Median & P95 & (\\%) & (\\%) & Lower & Not signif. & Lower & ETF signif. \\\\",
       " & & & & & & Sharpe (\\%) & (\\%) & Sharpe (\\%) & better (\\%)\\\\"], rows,
      f"500 random portfolios of $N$ stocks per row, drawn from the 500 most-traded NSE equities at the start of the "
      f"out-of-sample period (survivorship-bias free), equally weighted, rebalanced annually, SIP of Rs~5,000 per month, "
      f"whole shares, net of costs and taxes. P5/P95 are the 5th/95th percentiles across draws; volatility and MDD are medians. "
      f"Benchmark Sharpe ratios: Nifty 50 ETF {e['Nifty 50 ETF Sharpe']:.2f}, multi-asset $1/N$ ETF {e['1/N ETF Sharpe']:.2f}. "
      f"``Not signif.'' and ``ETF signif. better'' report the share of draws for which the \\citet{{ledoitwolf2008}} test does or "
      f"does not reject equal Sharpe ratios at 5\\%.")

# ---------------------------------------------------------------- Table 6: decomposition
t16 = rd("T16_risk_decomposition")
lab = []
for p in t16["portfolio"]:
    q = p.replace("(frictionless reference)", "").replace("(median of draws)", "(median)").strip()
    q = q.replace("All 100 top-100 stocks, equal-weight", "All top-100 stocks, equal weight$^{a}$")
    q = q.replace("All 500 top-500 stocks, equal-weight", "All top-500 stocks, equal weight$^{a}$")
    lab.append(q)
rows = [[esc(l).replace("\\$", "$"), pct(r["ann_return"]), pct(r["ann_vol"]), num(r["sharpe"]), pct(r["max_drawdown"]), num(r["beta vs Nifty 50 ETF"])]
        for l, (_, r) in zip(lab, t16.iterrows())]
write("tab_decomp", "Decomposing the excess risk of direct-equity portfolios", "tab:decomp", "lrrrrr",
      ["Portfolio & Return (\\%) & Vol. (\\%) & Sharpe & MDD (\\%) & $\\beta$ vs. Nifty 50 ETF \\\\"], rows,
      "SIP investor, Rs~5,000 per month, January 2018 to September 2026. $^{a}$Equal-weighted basket of every stock in the universe "
      "at the start date, frictionless and fractional (not investable by a small investor); it isolates the systematic difference "
      "between equal-weighting liquid stocks and the cap-weighted index. Random $N$-stock rows report medians over 500 draws, net "
      "of costs and taxes. The top-100 universe ranks stocks by three-month average traded value.")

# ---------------------------------------------------------------- Table 7: investor size and broker
t11 = rd("T11_erosion_by_investor")
rows = []
for _, r in t11.iterrows():
    rows.append([esc(r["investor"]), "Zero" if r["broker"] == "zero" else "Rs~20 flat", num(r["1/N ETF Sharpe"]), num(r["10-stock median Sharpe"]),
                 num(r["1/N ETF cost %"], 2), num(r["10-stock cost % (median)"], 2), num(r["1/N ETF tax % of gains"], 1),
                 num(r["10-stock tax % of gains (median)"], 1), pct(r["1/N ETF XIRR post-tax"]), pct(r["10-stock XIRR median"]),
                 pct(r["10-stock cash weight (median)"], 1)])
write("tab_investor", "Costs, taxes and performance by investor size and brokerage plan", "tab:investor", "llrrrrrrrrr",
      ["& & \\multicolumn{2}{c}{Sharpe} & \\multicolumn{2}{c}{Costs (\\% contrib.)} & \\multicolumn{2}{c}{Tax (\\% gains)} & \\multicolumn{2}{c}{XIRR (\\%)} & Cash \\\\",
       "\\cmidrule{3-4}\\cmidrule{5-6}\\cmidrule{7-8}\\cmidrule{9-10}",
       "Investor & Brokerage & ETF & Stocks & ETF & Stocks & ETF & Stocks & ETF & Stocks & drag (\\%)\\\\"], rows,
      "ETF is the multi-asset $1/N$ ETF portfolio; Stocks is the median random 10-stock portfolio (100 draws; 500 for the main investor). "
      "Under flat brokerage each SIP instalment is placed as a single order in the most underweight holding. Tax is total capital-gains "
      "tax as a percentage of total gains; cash drag is the median average share of the portfolio held as uninvested cash because "
      "only whole units can be bought. 20\\% income-tax slab.", size="\\scriptsize", sideways=True)

# ---------------------------------------------------------------- Appendix tables
t3 = rd("T03_correlations").set_index("key")
t3 = t3.loc[ETF_ORDER, ETF_ORDER]
rows = [[e] + [num(t3.loc[e, c]) if j <= i else "" for j, c in enumerate(ETF_ORDER)] for i, e in enumerate(ETF_ORDER)]
write("tab_corr", "Correlation matrix of monthly ETF returns", "tab:corr", "lrrrrr", ["& " + " & ".join(ETF_ORDER) + " \\\\"], rows,
      "Pearson correlations of monthly returns, February 2015 to September 2026.")

t6 = rd("T06_weights_diversification").set_index("strategy")
rows = [[STRAT_TEX[s]] + [pct(t6.loc[s, f"avg_w_{e}"], 1) for e in ETF_ORDER] + [num(t6.loc[s, "avg_diversification_ratio"])] for s in STRAT_ORDER]
write("tab_weights", "Average target weights and diversification ratio", "tab:weights", "lrrrrrr",
      ["Strategy & " + " & ".join(ETF_ORDER) + " & DR \\\\"], rows,
      "Average of target weights (\\%) over the annual rebalancing dates. DR is the diversification ratio "
      "(weighted average volatility over portfolio volatility) using the full-sample Ledoit--Wolf covariance matrix.")

t13 = rd("T13_subperiods")
t13["strategy"] = t13["strategy"].str.replace(r"Random 10-stock \(cross-sectional median monthly return\)", "Random 10 stocks$^{a}$", regex=True)
periods = list(dict.fromkeys(t13["subperiod"]))
strats = STRAT_ORDER + ["Random 10 stocks$^{a}$"]
rows = []
for s in strats:
    rr = [STRAT_TEX.get(s, s)]
    for p in periods:
        x = t13[(t13["subperiod"] == p) & (t13["strategy"] == s)]
        rr.append(pct(x["ann_return"].iloc[0]) + " / " + num(x["sharpe"].iloc[0]) if len(x) else "--")
    rows.append(rr)
write("tab_subperiods", "Sub-period performance: annualised return (\\%) / Sharpe ratio", "tab:subperiods", "l" + "r" * len(periods),
      ["Strategy & " + " & ".join(p.replace("and", "\\&") for p in periods) + " \\\\"], rows,
      "SIP investor, net of costs and taxes. Pre-COVID: Jan 2018--Jan 2020; COVID: Feb 2020--Dec 2021; then calendar years. "
      "$^{a}$Cross-sectional median monthly return across the random 10-stock portfolios.", size="\\scriptsize")

t14 = rd("T14_robustness")
rows = []
prev = None
for _, r in t14.iterrows():
    chk = r["check"] if r["check"] != prev else ""
    prev = r["check"]
    rows.append([esc(chk), esc(r["variant"]), STRAT_TEX.get(r["strategy"], esc(r["strategy"])), num(r["sharpe"]), pct(r["ann_return"]),
                 pct(r["max_drawdown"]), pct(r["xirr_post_tax"]), pct(r["cost_pct_contrib"], 2)])
write("tab_robust", "Robustness checks", "tab:robust", "lllrrrrr",
      ["Check & Variant & Strategy & Sharpe & Return (\\%) & MDD (\\%) & XIRR (\\%) & Costs (\\%) \\\\"], rows,
      "SIP investor (Rs~5,000 per month), net of costs and taxes unless stated. Window checks use a common out-of-sample start "
      "for comparability. The historical tax regime applies the capital-gains rules in force on each transaction date, "
      "including 31 January 2018 grandfathering.", size="\\scriptsize", sideways=True)

t15 = rd("T15_largecap_universe").set_index("N stocks")
rows = [[str(N), num(r["Sharpe p5"]), num(r["Sharpe median"]), num(r["Sharpe p95"]), pct(r["vol median"]), pct(r["MDD median"]),
         pct(r["XIRR median"]), f"{r['% draws below Nifty 50 ETF Sharpe']:.1f}", f"{r['% draws with higher vol than Nifty 50 ETF']:.1f}",
         f"{r['% draws with deeper MDD than Nifty 50 ETF']:.1f}"] for N, r in t15.iterrows()]
write("tab_largecap", "Random portfolios drawn from the 100 most-traded stocks", "tab:largecap", "rrrrrrrrrr",
      ["$N$ & P5 & Median & P95 & Vol. (\\%) & MDD (\\%) & XIRR (\\%) & Lower Sharpe (\\%) & Higher vol. (\\%) & Deeper MDD (\\%) \\\\"], rows,
      "As Table~\\ref{tab:stocks} but drawing from the 100 most-traded stocks (a Nifty 100-like universe). The last three columns "
      "compare each draw with the Nifty 50 ETF.")

t12 = rd("T12_slab_scenarios").set_index("strategy")
rows = [[STRAT_TEX[s]] + [num(t12.loc[s, f"sharpe | slab {k}"]) for k in ("5%", "20%", "30%")]
        + [pct(t12.loc[s, f"tax_pct_contrib | slab {k}"]) for k in ("5%", "20%", "30%")]
        + [pct(t12.loc[s, f"xirr_post_tax | slab {k}"]) for k in ("5%", "20%", "30%")] for s in STRAT_ORDER]
write("tab_slab", "Income-tax slab scenarios", "tab:slab", "lrrrrrrrrr",
      ["& \\multicolumn{3}{c}{Sharpe} & \\multicolumn{3}{c}{Tax (\\% contrib.)} & \\multicolumn{3}{c}{XIRR (\\%)} \\\\",
       "\\cmidrule{2-4}\\cmidrule{5-7}\\cmidrule{8-10}",
       "Strategy & 5\\% & 20\\% & 30\\% & 5\\% & 20\\% & 30\\% & 5\\% & 20\\% & 30\\% \\\\"], rows,
      "The slab rate applies to gains on debt ETFs and to short-term gains on gold and international ETFs; equity gains "
      "are taxed at flat rates. SIP investor, zero brokerage.")
