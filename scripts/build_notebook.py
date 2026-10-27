"""Assemble notebooks/ETF_SmallInvestor_India.ipynb from the cell definitions below and the src/ modules.

Run from the repository root:  python scripts/build_notebook.py
The notebook embeds the three modules so it runs stand-alone in Google Colab."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
SRC = {n: open(os.path.join(HERE, "src", n)).read() for n in ("etf_engine.py", "data_io.py", "study.py")}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================================
md(r"""
# Do Index ETFs Help Small Investors Diversify?
## Evidence from India with Realistic Costs, Taxes and Whole-Unit Purchases

*Kalaiarasan G (Amrita School of Artificial Intelligence) and A. V. Shyam (Amrita School of Business), Amrita Vishwa Vidyapeetham, Coimbatore, India*

Replication notebook for the journal paper. It downloads **free data only**, builds survivorship-bias-free
random stock portfolios, simulates realistic small investors (lump sum and **SIP**, whole-unit purchases,
**Indian transaction costs and capital-gains tax**), and saves every table, figure, configuration, data
manifest and reference needed for the manuscript.

**How to run in Google Colab**
1. `Runtime → Run all`. Allow Google Drive access when asked; everything is saved to
   `MyDrive/ETF_SmallInvestor_India/`.
2. First run downloads data (≈ 20–40 min: ~140 NSE bhavcopy files plus Yahoo prices). Later runs reuse the cache.
3. Set `QUICK_RUN = True` in Section 1 for a fast first pass (fewer random draws and bootstrap replications).
   Use `QUICK_RUN = False` for the numbers reported in the paper.
4. `DATA_MODE = "synthetic"` runs the full pipeline on simulated data in a few minutes. It is a
   smoke test only and must never be used for results.

**Project folder (created automatically)**
```
ETF_SmallInvestor_India/
├── README.md                 how the project is organised and how to reproduce it
├── config/                   config.json (latest) + config_<run>.json (every run)
├── src/                      etf_engine.py, data_io.py, study.py (written by this notebook)
├── data/raw/                 untouched downloads (cache): etf_prices, etf_nav, index_tri,
│                             risk_free, bhavcopy, stock_prices
├── data/processed/           cleaned panels used in the analysis
├── data/download_manifest.jsonl   URL, file, time, SHA-256 and rows of every download
├── references/               references.bib, data_sources.md, tax_cost_rules.md,
│                             methodology_assumptions.md
├── results/run_<timestamp>/  tables/ (CSV, LaTeX, all_tables.xlsx), figures/ (PNG 300 dpi, PDF),
│                             logs/ (run log, package versions), key_results.md, output_index.csv
└── manuscript/               notes for writing (results summary copied here)
```

**Sections**
1. Settings · 2. Environment and folders · 3. Source code · 4. Configuration · 5. References and documentation ·
6. Data acquisition · 7. Data preparation · 8. Descriptive statistics · 9. ETF tracking quality and liquidity ·
10. Dynamic correlations · 11. Strategy weights · 12. ETF portfolio simulations · 13. Random direct-equity portfolios ·
14. Statistical tests · 15. Cost and tax erosion · 16. Sub-periods · 17. Robustness · 18. Additional figures ·
19. Export and key results
""")

md("## 1. Settings")
code(r"""
DATA_MODE = "live"        # "live" = free data download | "synthetic" = smoke test only (never for results)
QUICK_RUN = False         # True = fewer random draws / bootstrap reps for a fast first pass
USE_DRIVE = True          # save everything to Google Drive (recommended)
PROJECT_NAME = "ETF_SmallInvestor_India"
AUTHOR = "Kalaiarasan G; A. V. Shyam"   # written into the configuration and README
FORCE_DOWNLOAD = False    # True = ignore the cache and re-download all data
""")

md("## 2. Environment and project folders")
code(r"""
import os, sys, json, time, platform, subprocess, datetime as dt, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance", "arch", "openpyxl"], check=False)

if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = f"/content/drive/MyDrive/{PROJECT_NAME}"
elif IN_COLAB:
    BASE = f"/content/{PROJECT_NAME}"
else:
    BASE = os.environ.get("ETF_BASE", os.path.abspath(PROJECT_NAME))

RUN_ID = dt.datetime.now().strftime("%Y%m%d_%H%M") + ("_synthetic" if DATA_MODE == "synthetic" else "") + ("_quick" if QUICK_RUN else "")
RUN_DIR = os.path.join(BASE, "results", f"run_{RUN_ID}")
DIRS = {
    "base": BASE, "src": f"{BASE}/src", "config": f"{BASE}/config", "references": f"{BASE}/references",
    "manuscript": f"{BASE}/manuscript", "raw": f"{BASE}/data/raw", "processed": f"{BASE}/data/processed",
    "raw_etf": f"{BASE}/data/raw/etf_prices", "raw_nav": f"{BASE}/data/raw/etf_nav",
    "raw_tri": f"{BASE}/data/raw/index_tri", "raw_rf": f"{BASE}/data/raw/risk_free",
    "raw_bhav": f"{BASE}/data/raw/bhavcopy", "raw_stock": f"{BASE}/data/raw/stock_prices",
    "results": RUN_DIR, "tables": f"{RUN_DIR}/tables", "figures": f"{RUN_DIR}/figures", "logs": f"{RUN_DIR}/logs",
}
if DATA_MODE == "synthetic":   # keep synthetic data away from real data
    DIRS["processed"] = f"{BASE}/data/processed_synthetic"
for p in DIRS.values():
    os.makedirs(p, exist_ok=True)
T0 = time.time()
print("Project folder:", BASE)
print("This run      :", RUN_DIR)
""")

md("## 3. Source code\nThe three modules are written to `src/` so the exact code used for every run is archived with the project.")
for name in ("etf_engine.py", "data_io.py", "study.py"):
    code(f"open(os.path.join(DIRS['src'], '{name}'), 'w').write(r'''{SRC[name]}''')\nprint('wrote src/{name}')")
code(r"""
import importlib
sys.path.insert(0, DIRS["src"])
import etf_engine as E, data_io as D, study as S
for m in (E, D, S):
    importlib.reload(m)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
S.set_style()
OUT = S.Outputs(DIRS)
MAN = D.Manifest(os.path.join(BASE, "data", "download_manifest.jsonl"))
""")

md("## 4. Configuration\nEvery assumption lives here and is saved to `config/`. Cost and tax parameters carry their sources.")
code(r"""
CONFIG = {
  "study": {
    "title": "Do Index ETFs Help Small Investors Diversify? Evidence from India with Realistic Costs, Taxes and Whole-Unit Purchases",
    "author": AUTHOR, "run_id": RUN_ID, "data_mode": DATA_MODE, "quick_run": QUICK_RUN},
  "period": {"start": "2015-01-01", "end": "2026-09-30"},
  # One ETF per asset class. Candidates are tried in order; the most liquid one with the
  # longest history passing the liquidity screen is kept (see Table 1).
  "etf_sleeves": {
    "eq_large":  {"label": "Nifty 50",      "type": "eq_etf",   "candidates": ["NIFTYBEES.NS", "SETFNIF50.NS", "NIFTYIETF.NS"], "tri_index": "NIFTY 50",      "min_traded_value_cr": 0.5},
    "eq_next50": {"label": "Nifty Next 50", "type": "eq_etf",   "candidates": ["JUNIORBEES.NS", "SETFNN50.NS"],                "tri_index": "NIFTY NEXT 50", "min_traded_value_cr": 0.2},
    "gold":      {"label": "Gold",          "type": "gold_etf", "candidates": ["GOLDBEES.NS", "HDFCGOLD.NS", "SETFGOLD.NS"],   "tri_index": None,            "min_traded_value_cr": 0.2},
    "debt":      {"label": "G-Sec",         "type": "debt_etf", "candidates": ["LTGILTBEES.NS", "SETF10GILT.NS", "LICNETFGSC.NS", "GILT5YBEES.NS"], "tri_index": None, "min_traded_value_cr": 0.005},
    "intl":      {"label": "Nasdaq 100",    "type": "intl_etf", "candidates": ["MON100.NS", "HNGSNGBEES.NS"],                 "tri_index": None,            "min_traded_value_cr": 0.1},
  },
  # Force a specific ticker for a sleeve (e.g. {"debt": "LTGILTBEES.NS"}) - used for robustness runs.
  "etf_override": {},
  # Return series per sleeve: "price" (market price, default) or "nav" (AMFI NAV, unit-normalised).
  # The G-Sec ETF trades thinly (stale prices, ~28% zero-volume days in the quick run), so its NAV is used;
  # Section 17 reports the market-price version as a robustness check.
  "price_source": {"debt": "nav"},
  # mfapi.in search terms used to find each ETF's NAV history (AMFI data)
  "nav_search": {
    "NIFTYBEES.NS":  {"query": "Nippon India ETF Nifty 50 BeES", "must": ["nifty 50", "bees"]},
    "SETFNIF50.NS":  {"query": "SBI Nifty 50 ETF", "must": ["nifty 50", "etf"]},
    "JUNIORBEES.NS": {"query": "Nippon India ETF Nifty Next 50 Junior BeES", "must": ["next 50", "bees"]},
    "SETFNN50.NS":   {"query": "SBI Nifty Next 50 ETF", "must": ["next 50", "etf"]},
    "GOLDBEES.NS":   {"query": "Nippon India ETF Gold BeES", "must": ["gold", "bees"]},
    "HDFCGOLD.NS":   {"query": "HDFC Gold ETF", "must": ["gold", "etf"]},
    "SETFGOLD.NS":   {"query": "SBI Gold ETF", "must": ["gold", "etf"]},
    "LTGILTBEES.NS": {"query": "Nippon India ETF Nifty 8-13 yr G-Sec Long Term Gilt", "must": ["gilt"]},
    "SETF10GILT.NS": {"query": "SBI Nifty 10 yr Benchmark G-Sec ETF", "must": ["g-sec", "etf"]},
    "LICNETFGSC.NS": {"query": "LIC MF Nifty 8-13 yr G-Sec ETF", "must": ["g-sec", "etf"]},
    "GILT5YBEES.NS": {"query": "Nippon India ETF Nifty 5 yr Benchmark G-Sec", "must": ["5 yr", "g-sec"]},
    "MON100.NS":     {"query": "Motilal Oswal Nasdaq 100 ETF", "must": ["nasdaq 100", "etf"]},
    "HNGSNGBEES.NS": {"query": "Nippon India ETF Hang Seng BeES", "must": ["hang seng", "bees"]},
  },
  "risk_free": {"fred_series": "INTGSTINM193N",
                "fallback_liquid_fund": {"query": "Nippon India Liquid Fund", "must": ["liquid", "direct", "growth"]}},
  "clean_spikes": True,     # fix bad ticks in ETF daily prices/NAVs and one-month spike-reversals in stock prices
  "stocks": {"large_cap_top_n": 100, "top_n": 500, "min_price": 10.0, "lookback_months": 3, "min_yahoo_coverage": 0.9,
             "N_list": [5, 10, 15, 20], "draws": 40 if QUICK_RUN else 500,
             "draws_grid": 15 if QUICK_RUN else 100, "seed": 2026},
  "backtest": {"window": 36, "rebalance": "A", "weight_cap": 0.40,
               "strategies": ["EQ100", "EQ_EW", "EW", "MinVar", "MaxSharpe", "RiskParity"], "main_cov": "lw",
               "cov_methods": ["sample", "lw", "dcc"], "gamma": 1.0},
  "labels": {"EQ100": "Nifty 50 ETF only", "EQ_EW": "Equity ETFs 1/N", "EW": "1/N ETF", "MinVar": "Min-variance ETF",
             "MaxSharpe": "Max-Sharpe ETF", "RiskParity": "Risk-parity ETF"},
  "investors": {"lump": [10000, 50000, 100000], "sip": [1000, 5000, 10000]},
  "main_investor": {"mode": "sip", "amount": 5000},
  "main_lump": {"mode": "lump", "amount": 50000},
  "brokers": ["zero", "flat"], "main_broker": "zero",
  # With a flat fee per order, a small SIP investor puts each month's instalment into the single most
  # underweight holding (one order) instead of buying every holding; full rebalancing still trades all.
  "sip_single_order_brokers": ["flat"],
  "slabs": [0.05, 0.20, 0.30], "main_slab": 0.20,
  "costs": {
    "flat_fee": 20.0,
    "stt": {"stock": [0.001, 0.001], "eq_etf": [0.0, 0.00001], "gold_etf": [0.0, 0.0],
            "debt_etf": [0.0, 0.0], "intl_etf": [0.0, 0.0]},
    "exchange": 0.0000297, "sebi": 0.000001, "stamp_buy": 0.00015, "gst": 0.18,
    "dp_charge_per_sell": 15.93,
    "half_spread": {"stock": 0.0010, "eq_etf": 0.0005, "gold_etf": 0.0010, "debt_etf": 0.0015, "intl_etf": 0.0020},
    "notes": {
      "stt": "Delivery equity 0.1% buy and sell; equity ETF units 0.001% on sale only; non-equity ETFs nil. NSE: https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies",
      "stamp_buy": "0.015% on delivery purchases, paid by buyer (Indian Stamp Act amendment, 1 July 2020). Source as above.",
      "exchange": "NSE cash-market transaction charge 0.00297% (uniform rate). Verify on the NSE charges page before submission.",
      "sebi": "SEBI turnover fee Rs 10 per crore.",
      "gst": "18% on brokerage, exchange and SEBI fees.",
      "dp_charge_per_sell": "Depository + broker DP charge per scrip per sell day, incl. GST (typical discount broker). Verify for the broker you describe.",
      "flat_fee": "Rs 20 per executed order (typical discount broker intraday/F&O fee; several brokers charge it on delivery too).",
      "half_spread": "Assumed half bid-ask spread; robustness adds 10/25/50 bps."}},
  "tax": {
    "regime": "current", "cess": 0.04, "eq_st_rate": 0.20, "eq_lt_rate": 0.125, "eq_lt_exemption": 125000.0,
    "oth_lt_rate": 0.125, "payment_month": 7,
    "cii": {2014: 240, 2015: 254, 2016: 264, 2017: 272, 2018: 280, 2019: 289, 2020: 301, 2021: 317,
            2022: 331, 2023: 348, 2024: 363, 2025: 376},
    "notes": {
      "equity": "Stocks and equity ETFs: STCG 20% (Sec 111A) if held <= 12 months; LTCG 12.5% above Rs 1.25 lakh per FY aggregate (Sec 112A). ClearTax: https://cleartax.in/s/how-are-etfs-taxed",
      "gold_intl": "Gold and international ETFs: STCG at slab rate (<= 12 months); LTCG 12.5% without indexation, no Rs 1.25 lakh exemption. IIFL: https://www.iifl.com/blogs/gold-loan/gold-etf-tax-in-india-2026-post-finance-act-2024-rules",
      "debt": "Debt ETFs bought on/after 1 April 2023: slab rate regardless of holding period (Sec 50AA). Finnovate: https://www.finnovate.in/learn/blog/etf-taxation-india",
      "unchanged_2026": "Structure set by Finance (No.2) Act 2024 and unchanged in Budget 2026. https://finlecture.in/indian-tax-system/capital-gains-tax-fy-2026-27/",
      "historical_regime": "Robustness only: equity LTCG exempt before 1-Apr-2018, 10% above Rs 1 lakh until 22-Jul-2024 with 31-Jan-2018 grandfathering; equity STCG 15% until 22-Jul-2024; non-equity LTCG 20% with indexation after 36 months for sales before 23-Jul-2024; Sec 50AA slab taxation for units bought from 1-Apr-2023 (gold/international until 31-Mar-2025). CII values should be verified against CBDT notifications.",
      "simplifications": "Resident individual; 4% cess; no surcharge; tax paid each July from the portfolio; losses set off and carried forward; dividends ignored (growth ETFs)."}},
  "tests": {"bootstrap_M": 499 if QUICK_RUN else 4999, "stock_test_M": 199 if QUICK_RUN else 999,
            "block": 5, "alpha": 0.05, "seed": 42},
  "robustness": {"windows": [36, 48, 60], "rebalance": ["M", "Q", "A", "T5"], "extra_bps": [10, 25, 50]},
  "subperiods": {"Pre-COVID": ["2015-01", "2020-01"], "COVID crash and recovery": ["2020-02", "2021-12"],
                 "2022 rate hikes": ["2022-01", "2022-12"], "2023-2024": ["2023-01", "2024-12"],
                 "2025 tariff shock": ["2025-01", "2025-12"], "2026 to date": ["2026-01", "2026-12"]},
}
with open(f"{DIRS['config']}/config.json", "w") as f: json.dump(CONFIG, f, indent=2)
with open(f"{DIRS['config']}/config_{RUN_ID}.json", "w") as f: json.dump(CONFIG, f, indent=2)
with open(f"{DIRS['logs']}/config_used.json", "w") as f: json.dump(CONFIG, f, indent=2)
LBL = CONFIG["labels"]
print("Configuration saved.")
""")

md("## 5. References and documentation\nWrites the bibliography, data-source register, cost/tax rule sheet, assumptions log and README.")
code(r'''
TODAY = dt.date.today().isoformat()
BIB = r"""
@article{markowitz1952, author={Markowitz, Harry}, title={Portfolio Selection}, journal={Journal of Finance}, volume={7}, number={1}, pages={77--91}, year={1952}}
@article{tobin1958, author={Tobin, James}, title={Liquidity Preference as Behavior towards Risk}, journal={Review of Economic Studies}, volume={25}, number={2}, pages={65--86}, year={1958}}
@article{sharpe1964, author={Sharpe, William F.}, title={Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk}, journal={Journal of Finance}, volume={19}, number={3}, pages={425--442}, year={1964}}
@article{evansarcher1968, author={Evans, John L. and Archer, Stephen H.}, title={Diversification and the Reduction of Dispersion: An Empirical Analysis}, journal={Journal of Finance}, volume={23}, number={5}, pages={761--767}, year={1968}}
@article{fama1970, author={Fama, Eugene F.}, title={Efficient Capital Markets: A Review of Theory and Empirical Work}, journal={Journal of Finance}, volume={25}, number={2}, pages={383--417}, year={1970}}
@article{samuelson1974, author={Samuelson, Paul A.}, title={Challenge to Judgment}, journal={Journal of Portfolio Management}, volume={1}, number={1}, pages={17--19}, year={1974}}
@article{brinson1986, author={Brinson, Gary P. and Hood, L. Randolph and Beebower, Gilbert L.}, title={Determinants of Portfolio Performance}, journal={Financial Analysts Journal}, volume={42}, number={4}, pages={39--44}, year={1986}}
@article{statman1987, author={Statman, Meir}, title={How Many Stocks Make a Diversified Portfolio?}, journal={Journal of Financial and Quantitative Analysis}, volume={22}, number={3}, pages={353--363}, year={1987}}
@article{sharpe1991, author={Sharpe, William F.}, title={The Arithmetic of Active Management}, journal={Financial Analysts Journal}, volume={47}, number={1}, pages={7--9}, year={1991}}
@article{barberodean2000, author={Barber, Brad M. and Odean, Terrance}, title={Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors}, journal={Journal of Finance}, volume={55}, number={2}, pages={773--806}, year={2000}}
@article{engle2002, author={Engle, Robert}, title={Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models}, journal={Journal of Business and Economic Statistics}, volume={20}, number={3}, pages={339--350}, year={2002}}
@article{ledoitwolf2004, author={Ledoit, Olivier and Wolf, Michael}, title={Honey, I Shrunk the Sample Covariance Matrix}, journal={Journal of Portfolio Management}, volume={30}, number={4}, pages={110--119}, year={2004}}
@article{romanowolf2005, author={Romano, Joseph P. and Wolf, Michael}, title={Stepwise Multiple Testing as Formalized Data Snooping}, journal={Econometrica}, volume={73}, number={4}, pages={1237--1282}, year={2005}}
@article{french2008, author={French, Kenneth R.}, title={The Cost of Active Investing}, journal={Journal of Finance}, volume={63}, number={4}, pages={1537--1573}, year={2008}}
@article{goetzmannkumar2008, author={Goetzmann, William N. and Kumar, Alok}, title={Equity Portfolio Diversification}, journal={Review of Finance}, volume={12}, number={3}, pages={433--463}, year={2008}}
@article{ledoitwolf2008, author={Ledoit, Olivier and Wolf, Michael}, title={Robust Performance Hypothesis Testing with the Sharpe Ratio}, journal={Journal of Empirical Finance}, volume={15}, number={5}, pages={850--859}, year={2008}}
@article{demiguel2009, author={DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman}, title={Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?}, journal={Review of Financial Studies}, volume={22}, number={5}, pages={1915--1953}, year={2009}}
@article{ledoitwolf2011, author={Ledoit, Olivier and Wolf, Michael}, title={Robust Performance Hypothesis Testing with the Variance}, journal={Wilmott}, volume={2011}, number={55}, pages={86--89}, year={2011}}
@article{baumanntrautmann2013, author={Baumann, Philipp and Trautmann, Norbert}, title={Portfolio-Optimization Models for Small Investors}, journal={Mathematical Methods of Operations Research}, volume={77}, number={3}, pages={345--356}, year={2013}}
@article{andrews1991, author={Andrews, Donald W. K.}, title={Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation}, journal={Econometrica}, volume={59}, number={3}, pages={817--858}, year={1991}}
@article{singhkaur2016, author={Singh, Jaspal and Kaur, Parneet}, title={Tracking Efficiency of Exchange Traded Funds (ETFs): Empirical Evidence from Indian Equity ETFs}, journal={Paradigm}, volume={20}, number={2}, pages={176--190}, year={2016}}
@article{tripathisethi2022, author={Tripathi, Vanita and Sethi, Aakanksha}, title={An Evaluation of the Tracking Performance of Exchange Traded Funds (ETFs): The Case of Indian Index ETFs}, journal={Vision: The Journal of Business Perspective}, year={2022}, doi={10.1177/0972262921996485}}
@techreport{rajuagarwalla2021, author={Raju, Rajan and Agarwalla, Sobhesh Kumar}, title={Equity Portfolio Diversification: How Many Stocks Are Enough? Evidence from India}, institution={Indian Institute of Management Ahmedabad}, type={Working Paper}, year={2021}, url={https://www.iima.ac.in/publication/equity-portfolio-diversification-how-many-stocks-are-enough-evidence-india}}
@misc{spivaindia2025, author={{S\&P Dow Jones Indices}}, title={SPIVA India Scorecard Year-End 2025}, year={2026}, url={https://www.spglobal.com/spdji/en/spiva/article/spiva-india-year-end-2025/}}
@misc{data_yahoo, author={{Yahoo Finance}}, title={Historical prices via the yfinance Python package}, url={https://pypi.org/project/yfinance/}, note={Accessed TODAY}}
@misc{data_nse_bhav, author={{National Stock Exchange of India}}, title={Capital Market Bhavcopy Archives}, url={https://www.nseindia.com/all-reports}, note={Accessed TODAY}}
@misc{data_amfi, author={{Association of Mutual Funds in India}}, title={NAV history (via mfapi.in)}, url={https://www.mfapi.in/}, note={Accessed TODAY}}
@misc{data_fred, author={{Federal Reserve Bank of St. Louis}}, title={Interest Rates, Government Securities, Treasury Bills for India (INTGSTINM193N)}, url={https://fred.stlouisfed.org/series/INTGSTINM193N}, note={Accessed TODAY}}
@misc{data_niftyindices, author={{NSE Indices Ltd}}, title={Total Return Index historical data}, url={https://www.niftyindices.com/reports/historical-data}, note={Accessed TODAY}}
@misc{nse_levies, author={{National Stock Exchange of India}}, title={Turnover fees, STT and other levies}, url={https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies}, note={Accessed TODAY}}
@misc{cleartax_etf, author={{ClearTax}}, title={How ETFs Are Taxed in India (FY 2025-26)}, url={https://cleartax.in/s/how-are-etfs-taxed}, note={Accessed TODAY}}
""".replace("TODAY", TODAY)
open(f"{DIRS['references']}/references.bib", "w").write(BIB.strip() + "\n")

open(f"{DIRS['references']}/data_sources.md", "w").write(f"""# Data sources (all free)

| Data | Source | Access | Use in study |
|---|---|---|---|
| ETF daily prices, volume | Yahoo Finance (`yfinance`) | Python package | Tradable prices, returns, liquidity |
| ETF daily NAV | AMFI NAV history via https://api.mfapi.in | JSON API | Tracking error, premium/discount |
| Index total return (TRI) | https://www.niftyindices.com (historical data) | JSON endpoint | Tracking error benchmark |
| All listed equities, month-end | NSE bhavcopy archives (nsearchives.nseindia.com) | ZIP files | Survivorship-free stock universe |
| Stock adjusted prices | Yahoo Finance; bhavcopy closes with split/bonus correction as fallback | | Random stock portfolios |
| Risk-free rate | FRED INTGSTINM193N (India 3-month T-bill); liquid-fund NAV return as fallback | CSV | Excess returns, Sharpe |

Every downloaded file is logged in `data/download_manifest.jsonl` with URL, UTC time, SHA-256 and row count.
First written {TODAY}.
""")

open(f"{DIRS['references']}/tax_cost_rules.md", "w").write(
    "# Transaction-cost and tax rules used\n\nGenerated from config.json on " + TODAY + ".\n\n## Costs\n\n" +
    "\n".join(f"- **{k}**: {v}" for k, v in CONFIG["costs"]["notes"].items()) +
    "\n\n## Taxes\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in CONFIG["tax"]["notes"].items()) +
    "\n\nCII (base 2001-02 = 100): " + json.dumps(CONFIG["tax"]["cii"]) + "\n")

open(f"{DIRS['references']}/methodology_assumptions.md", "w").write(f"""# Methodology and assumptions log

1. **Out-of-sample design.** Weights are estimated on a rolling {CONFIG['backtest']['window']}-month window using data up to the trade date only (DeMiguel et al., 2009). Trades execute at month-end closing prices.
2. **ETF strategies.** Nifty 50 ETF only; 1/N across ETFs; minimum-variance; constrained maximum-Sharpe; risk parity. Long-only, weight cap {CONFIG['backtest']['weight_cap']:.0%} for optimised portfolios. Covariance: Ledoit-Wolf shrinkage (main), sample and DCC-GARCH (robustness).
3. **Direct-equity portfolios.** N in {CONFIG['stocks']['N_list']} stocks drawn at random from the {CONFIG['stocks']['top_n']} most-traded NSE equities on the start date (3-month average traded value, price at least Rs {CONFIG['stocks']['min_price']:.0f}). Universe built from month-end bhavcopies, so later-delisted stocks are included. Equal weights, held to the end; delisted stocks sold at their last price and proceeds reinvested at the next trade.
4. **Small-investor frictions.** Whole units only; greedy allocation of available cash toward target weights; leftover cash earns zero. Costs: brokerage (zero or Rs 20 flat), STT, exchange, SEBI, stamp duty, GST, DP charge and half-spread.
5. **SIP.** Monthly contribution buys toward target weights without selling; full rebalancing annually (main case).
6. **Tax.** FIFO lots; FY (April-March) netting, loss set-off and carry-forward; tax paid each July from the portfolio; terminal liquidation taxed. Current FY2026-27 rules in the main case, historical date-dependent rules as robustness.
7. **Returns.** Time-weighted monthly returns net of costs and taxes paid; money-weighted XIRR on post-tax terminal wealth.
8. **Inference.** Ledoit-Wolf (2008) studentised circular block bootstrap for Sharpe differences, Ledoit-Wolf (2011) for variance differences, Romano-Wolf (2005) stepdown for multiple comparisons; block length {CONFIG['tests']['block']}.
9. **Known limitations.** Adjusted prices used as tradable prices; dividends on stocks reinvested via adjusted prices and not taxed separately; delisting returns set to last traded price; Yahoo coverage of delisted stocks is incomplete (bhavcopy fallback with split heuristic, share reported in the stock-universe table); expense ratios captured through ETF prices; ETF AUM not used in screening (liquidity screen on traded value instead).
""")

if not os.path.exists(f"{BASE}/README.md"):
    open(f"{BASE}/README.md", "w").write(f"""# {CONFIG['study']['title']}

Author: {AUTHOR or '(add name)'}

Reproduce: open `ETF_SmallInvestor_India.ipynb` in Google Colab and run all cells.
Raw downloads are cached in `data/raw/`; each run writes to `results/run_<timestamp>/`.

- `config/` every run's parameters
- `references/` bibliography (BibTeX), data sources, tax and cost rules, assumptions
- `data/download_manifest.jsonl` provenance of every downloaded file
- `results/run_*/tables` CSV + LaTeX + all_tables.xlsx; `figures` PNG (300 dpi) + PDF; `key_results.md`
""")
print("References and documentation written to", DIRS["references"])
''')

md("## 6. Data acquisition\nLive mode downloads from free sources and caches raw files; synthetic mode generates placeholder data with the same structure.")
code(r"""
P_DIR = DIRS["processed"]
cfg_p = CONFIG["period"]
SLEEVES = CONFIG["etf_sleeves"]

def cached(path):
    return os.path.exists(path) and not FORCE_DOWNLOAD

if DATA_MODE == "synthetic":
    syn = D.synthetic_dataset(list(SLEEVES), cfg_p["start"], cfg_p["end"])
    syn["etf_daily"].to_csv(f"{P_DIR}/etf_daily.csv.gz", index=False)
    syn["etf_nav"].to_csv(f"{P_DIR}/etf_nav.csv.gz", index=False)
    syn["etf_tri"].to_csv(f"{P_DIR}/etf_tri.csv.gz", index=False)
    syn["rf"].assign(source="synthetic").to_csv(f"{P_DIR}/rf_monthly.csv", index=False)
    syn["stock_prices"].to_csv(f"{P_DIR}/stock_prices_monthly.csv.gz")
    syn["stock_eligible"].to_csv(f"{P_DIR}/stock_eligible.csv.gz")
    syn["stock_rank"].to_csv(f"{P_DIR}/stock_rank.csv.gz")
    syn["stock_meta"].to_csv(f"{P_DIR}/stock_meta.csv", index=False)
    pd.DataFrame([{"sleeve": k, "label": v["label"], "ticker": f"SYN_{k}", "type": v["type"]} for k, v in SLEEVES.items()]
                 ).to_csv(f"{P_DIR}/etf_selection.csv", index=False)
    print("Synthetic data generated (smoke test only).")
""")
code(r"""
# ---- 6a. ETF prices (Yahoo Finance) and selection of one ETF per sleeve --------------------
if DATA_MODE == "live":
    raw_path = f"{DIRS['raw_etf']}/etf_candidates_daily.csv.gz"
    tickers = sorted({t for s in SLEEVES.values() for t in s["candidates"]})
    if cached(raw_path):
        etf_raw = pd.read_csv(raw_path, parse_dates=["date"])
    else:
        etf_raw = D.yf_history(tickers, cfg_p["start"], (pd.Timestamp(cfg_p["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        etf_raw.to_csv(raw_path, index=False)
        MAN.log("Yahoo Finance", "yfinance:" + ",".join(tickers), raw_path, len(etf_raw))
    rows = []
    for key, s in SLEEVES.items():
        for rank_, tk in enumerate(s["candidates"]):
            d = etf_raw[etf_raw["ticker"] == tk].dropna(subset=["close"])
            if d.empty:
                rows.append({"sleeve": key, "ticker": tk, "available": False}); continue
            d = d[d["volume"] > 0]
            last2y = d[d["date"] >= d["date"].max() - pd.Timedelta(days=730)]
            tv = (last2y["close"] * last2y["volume"]).median() / 1e7
            rows.append({"sleeve": key, "label": s["label"], "ticker": tk, "type": s["type"], "available": True,
                         "first_date": d["date"].min(), "last_date": d["date"].max(),
                         "median_daily_traded_value_cr_2y": tv, "passes_liquidity": tv >= s["min_traded_value_cr"],
                         "candidate_order": rank_})
    cand = pd.DataFrame(rows)
    cand.to_csv(f"{P_DIR}/etf_candidates.csv", index=False)
    ok = cand[cand["available"] & cand["passes_liquidity"]].sort_values(["sleeve", "first_date", "candidate_order"])
    sel = ok.groupby("sleeve").head(1)
    for slv, tk in CONFIG.get("etf_override", {}).items():
        forced = cand[(cand["sleeve"] == slv) & (cand["ticker"] == tk) & cand["available"]]
        if len(forced):
            sel = pd.concat([sel[sel["sleeve"] != slv], forced]).sort_values("sleeve")
            print(f"Override: {slv} -> {tk}")
    missing = set(SLEEVES) - set(sel["sleeve"])
    if missing:
        print("WARNING: no liquid ETF found for", missing, "- check tickers in CONFIG; these sleeves are dropped.")
    sel.to_csv(f"{P_DIR}/etf_selection.csv", index=False)
    etf_daily = etf_raw.merge(sel[["ticker", "sleeve"]], on="ticker").rename(columns={"sleeve": "key"})
    etf_daily[["date", "key", "close", "adj_close", "volume"]].to_csv(f"{P_DIR}/etf_daily.csv.gz", index=False)
    display(cand)
""")
code(r"""
# ---- 6b. ETF NAVs (AMFI via mfapi) and index TRI (niftyindices) ----------------------------
if DATA_MODE == "live":
    sel = pd.read_csv(f"{P_DIR}/etf_selection.csv")
    navs, maps, tris = [], [], []
    for _, r in sel.iterrows():
        f = f"{DIRS['raw_nav']}/{r['ticker']}.csv"
        spec = CONFIG["nav_search"].get(r["ticker"])
        if cached(f):
            nav = pd.read_csv(f, parse_dates=["date"]); maps.append({"ticker": r["ticker"], "scheme": "cached"})
        elif spec:
            item = D.pick_scheme(spec["query"], spec["must"], avoid=["fund of fund", "fof"])
            if item:
                nav = D.mfapi_nav(item["schemeCode"]); nav.to_csv(f, index=False)
                MAN.log("AMFI via mfapi", f"https://api.mfapi.in/mf/{item['schemeCode']}", f, len(nav), item["schemeName"])
                maps.append({"ticker": r["ticker"], "scheme_code": item["schemeCode"], "scheme": item["schemeName"]})
            else:
                nav = pd.DataFrame(columns=["date", "nav"]); maps.append({"ticker": r["ticker"], "scheme": "NOT FOUND"})
        else:
            nav = pd.DataFrame(columns=["date", "nav"])
        navs.append(nav.assign(key=r["sleeve"]))
        idx = SLEEVES[r["sleeve"]]["tri_index"]
        if idx:
            ft = f"{DIRS['raw_tri']}/{idx.replace(' ', '_')}_api.csv"
            tri = D.read_tri_files(DIRS["raw_tri"], idx)          # manual downloads (and earlier API cache)
            if len(tri):
                print(f"  TRI {idx}: {len(tri)} days from files in data/raw/index_tri ({tri['date'].min():%Y-%m-%d} to {tri['date'].max():%Y-%m-%d})")
            else:
                tri = D.nifty_tri(idx, cfg_p["start"], cfg_p["end"])
                if len(tri):
                    tri.to_csv(ft, index=False); MAN.log("niftyindices TRI", "niftyindices.com", ft, len(tri), idx)
                else:
                    print(f"  TRI for {idx} not available. To add it: niftyindices.com -> Reports -> Historical Data -> "
                          f"Total Returns Index, select {idx}, download each year as CSV and save to "
                          f"{DIRS['raw_tri']}/{idx.replace(' ', '_')}_<year>.csv, then re-run this cell.")
            tris.append(tri.assign(key=r["sleeve"]))
    pd.concat(navs).to_csv(f"{P_DIR}/etf_nav.csv.gz", index=False)
    (pd.concat(tris) if tris else pd.DataFrame(columns=["date", "tri", "key"])).to_csv(f"{P_DIR}/etf_tri.csv.gz", index=False)
    pd.DataFrame(maps).to_csv(f"{P_DIR}/etf_nav_scheme_map.csv", index=False)
    display(pd.DataFrame(maps))
""")
code(r"""
# ---- 6b-check. NAV coverage, price/NAV consistency and ETF trading activity --------------
if DATA_MODE == "live":
    sel = pd.read_csv(f"{P_DIR}/etf_selection.csv")
    ed = pd.read_csv(f"{P_DIR}/etf_daily.csv.gz", parse_dates=["date"])
    nv = pd.read_csv(f"{P_DIR}/etf_nav.csv.gz", parse_dates=["date"])
    chk = []
    for _, r in sel.iterrows():
        k = r["sleeve"]
        p = ed[ed["key"] == k].set_index("date")
        n_raw = nv[nv["key"] == k].set_index("date")["nav"]
        n = D.normalize_nav_units(p["adj_close"], n_raw)
        fixed_days = int((np.abs(n.reindex(n_raw.index) / n_raw - 1) > 1e-9).sum())
        ratio = (p["close"] / n.reindex(p.index)).dropna()
        yr = ratio.groupby(ratio.index.year).median()
        zero = (p["volume"] == 0).groupby(p.index.year).mean()
        chk.append({"sleeve": k, "ticker": r["ticker"],
                    "nav_first": n.index.min(), "nav_last": n.index.max(), "nav_days": len(n),
                    "price_days": len(p), "nav_coverage_of_price_days": len(ratio) / max(len(p), 1),
                    "NAV days unit-adjusted (split)": fixed_days,
                    "price/NAV median (min year)": yr.min() if len(yr) else np.nan,
                    "price/NAV median (max year)": yr.max() if len(yr) else np.nan,
                    "zero-volume days, worst year": zero.max(), "worst year": zero.idxmax() if len(zero) else None})
    CHECK = pd.DataFrame(chk).set_index("sleeve")
    CHECK.to_csv(f"{P_DIR}/etf_data_check.csv")
    display(CHECK)
    print("What to look for:\n"
          " - nav_coverage close to 1 and nav_first near the ETF's price history start; otherwise the mfapi match is wrong\n"
          "   (fix the search terms in CONFIG['nav_search'] or delete data/raw/etf_nav/<ticker>.csv and re-run 6b).\n"
          " - price/NAV medians (after the automatic unit-split fix) close to 1.00 in every year.\n"
          " - zero-volume share above ~10% in any year means stale prices; consider CONFIG['etf_override'].")
""")
code(r"""
# ---- 6c. Risk-free rate ------------------------------------------------------------------
if DATA_MODE == "live":
    f = f"{DIRS['raw_rf']}/fred_{CONFIG['risk_free']['fred_series']}.csv"
    if cached(f):
        fr = pd.read_csv(f, parse_dates=["date"])
    else:
        fr = D.fred_series(CONFIG["risk_free"]["fred_series"])
        if len(fr): fr.to_csv(f, index=False); MAN.log("FRED", CONFIG["risk_free"]["fred_series"], f, len(fr))
    months = pd.date_range(cfg_p["start"], cfg_p["end"], freq="ME")
    rf = pd.Series(np.nan, index=months)
    if len(fr):
        m = fr.set_index("date")["value"].resample("ME").last()
        rf = ((1 + m / 100) ** (1 / 12) - 1).reindex(months)
    src = pd.Series(np.where(rf.notna(), "FRED T-bill", "liquid fund NAV"), index=months)
    if rf.isna().any():
        f2 = f"{DIRS['raw_rf']}/liquid_fund_nav.csv"
        if cached(f2):
            lf = pd.read_csv(f2, parse_dates=["date"])
        else:
            spec = CONFIG["risk_free"]["fallback_liquid_fund"]
            item = D.pick_scheme(spec["query"], spec["must"])
            lf = D.mfapi_nav(item["schemeCode"]) if item else pd.DataFrame(columns=["date", "nav"])
            lf.to_csv(f2, index=False); MAN.log("AMFI via mfapi", "liquid fund", f2, len(lf), item["schemeName"] if item else "")
        lr = lf.set_index("date")["nav"].resample("ME").last().pct_change().reindex(months)
        rf = rf.fillna(lr)
    rf = rf.ffill().bfill()
    pd.DataFrame({"date": months, "rf": rf.values, "source": src.values}).to_csv(f"{P_DIR}/rf_monthly.csv", index=False)
    print(src.value_counts())
""")
code(r"""
# ---- 6d. Survivorship-free stock universe: NSE month-end bhavcopies + Yahoo adjusted prices --
if DATA_MODE == "live":
    bhav = D.month_end_bhavcopies(cfg_p["start"], cfg_p["end"], DIRS["raw_bhav"], MAN, force=FORCE_DOWNLOAD)
    if bhav.empty:
        raise RuntimeError("NSE archives unreachable from this runtime. Re-run later, or use "
                           "D.nifty500_current_list() as a (survivorship-biased) fallback and state it in the paper.")
    b = bhav[bhav["series"] == "EQ"].copy()
    b["id"] = b["isin"].fillna(b["symbol"])
    syms = sorted(b.sort_values("date").groupby("id")["symbol"].last().unique())
    f = f"{DIRS['raw_stock']}/yahoo_stocks_daily.csv.gz"
    if cached(f):
        ys = pd.read_csv(f, parse_dates=["date"])
    else:
        print(f"Downloading {len(syms)} stocks from Yahoo (several minutes)...")
        ys = D.yf_history([s + ".NS" for s in syms], cfg_p["start"],
                          (pd.Timestamp(cfg_p["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), batch=100)
        ys[["date", "ticker", "close", "adj_close", "volume"]].to_csv(f, index=False)
        MAN.log("Yahoo Finance", f"yfinance: {len(syms)} NSE stocks", f, len(ys))
    ym = ys.pivot_table(index="date", columns="ticker", values="adj_close").resample("ME").last()
    sc = CONFIG["stocks"]
    P, ELIG, META = D.build_stock_panel(bhav, ym, top_n=sc["top_n"], min_price=sc["min_price"],
                                        lookback=sc["lookback_months"], min_yf_coverage=sc["min_yahoo_coverage"])
    P.to_csv(f"{P_DIR}/stock_prices_monthly.csv.gz"); ELIG.to_csv(f"{P_DIR}/stock_eligible.csv.gz")
    D.LAST_RANK.to_csv(f"{P_DIR}/stock_rank.csv.gz")
    META.to_csv(f"{P_DIR}/stock_meta.csv", index=False)
    print(META["price_source"].value_counts())
""")

md("## 7. Data preparation")
code(r"""
etf_daily = pd.read_csv(f"{P_DIR}/etf_daily.csv.gz", parse_dates=["date"])
etf_nav = pd.read_csv(f"{P_DIR}/etf_nav.csv.gz", parse_dates=["date"])
etf_tri = pd.read_csv(f"{P_DIR}/etf_tri.csv.gz", parse_dates=["date"])
SEL = pd.read_csv(f"{P_DIR}/etf_selection.csv")
rf_df = pd.read_csv(f"{P_DIR}/rf_monthly.csv", parse_dates=["date"])
RF = rf_df.set_index("date")["rf"]; RF.index = RF.index.to_period("M").to_timestamp("M")
SP = pd.read_csv(f"{P_DIR}/stock_prices_monthly.csv.gz", index_col=0, parse_dates=True)
SE = pd.read_csv(f"{P_DIR}/stock_eligible.csv.gz", index_col=0, parse_dates=True).astype(bool)
SMETA = pd.read_csv(f"{P_DIR}/stock_meta.csv")
_rk = f"{P_DIR}/stock_rank.csv.gz"
SRANK = pd.read_csv(_rk, index_col=0, parse_dates=True) if os.path.exists(_rk) else None
if SRANK is not None:
    SRANK.index = SRANK.index.to_period("M").to_timestamp("M")
SP_RAW = SP.copy()
N_SPIKES = 0
if CONFIG.get("clean_spikes"):
    SP, N_SPIKES = D.clean_monthly_spikes(SP)
    print(f"Stock prices: {N_SPIKES} one-month spike-reversals fixed")
for X in (SP, SP_RAW, SE):
    X.index = X.index.to_period("M").to_timestamp("M")

KEYS = [k for k in SLEEVES if k in set(SEL["sleeve"])]
TYPES = {k: SLEEVES[k]["type"] for k in KEYS}
NAMES = {k: SLEEVES[k]["label"] for k in KEYS}
PXd = S.wide_daily(etf_daily, "key", "adj_close")[KEYS]
ETF_FIXED = {}
if CONFIG.get("clean_spikes"):
    for k in KEYS:
        PXd[k], ETF_FIXED[k] = D.despike_daily(PXd[k])
    print("ETF daily prices, bad ticks fixed:", ETF_FIXED)
PXd_MKT = PXd.copy()                       # market prices, kept for robustness
_navw = S.wide_daily(etf_nav, "key", "nav") if len(etf_nav) else pd.DataFrame()
NAVN = pd.DataFrame({k: D.normalize_nav_units(PXd[k], _navw[k].dropna()) for k in KEYS if k in _navw})
if CONFIG.get("clean_spikes"):
    for k in NAVN:
        NAVN[k], nfix = D.despike_daily(NAVN[k])
        ETF_FIXED[f"{k} NAV"] = nfix
for k, src in CONFIG.get("price_source", {}).items():
    if src == "nav" and k in NAVN and k in PXd:
        nav_k = NAVN[k].reindex(PXd.index.union(NAVN[k].index)).ffill().reindex(PXd.index)
        PXd[k] = nav_k.where(nav_k.notna(), PXd[k])
        print(f"{k}: returns from unit-normalised NAV ({nav_k.notna().mean():.0%} of trading days)")
PXm = PXd.resample("ME").last()
Rd = PXd.pct_change().dropna(how="all")
Rm = PXm.pct_change()
start_common = PXm.dropna().index[0]
Rm = Rm.loc[start_common:].dropna()
W = CONFIG["backtest"]["window"]
OOS = PXm.loc[start_common:].index[W:]
PX_OOS = PXm.loc[OOS]
RF_OOS = RF.reindex(OOS).ffill()
print(f"ETFs: {NAMES}")
print(f"Common data start: {start_common:%Y-%m}; out-of-sample: {OOS[0]:%Y-%m} to {OOS[-1]:%Y-%m} ({len(OOS)} months)")
""")

md("## 8. Descriptive statistics")
code(r"""
from scipy import stats as sst
t1 = SEL.copy()
if "first_date" in t1:
    t1["first_date"] = pd.to_datetime(t1["first_date"]).dt.date
OUT.table(t1.set_index("sleeve"), "T01_etf_selection", "ETFs selected for each asset class (free Yahoo Finance data; liquidity screen on median daily traded value, Rs crore).")

desc = []
for k in KEYS:
    r = Rm[k]
    jb = sst.jarque_bera(r)
    desc.append({"ETF": NAMES[k], "mean_monthly_%": r.mean() * 100, "sd_monthly_%": r.std() * 100,
                 "ann_return_%": ((1 + r).prod() ** (12 / len(r)) - 1) * 100, "ann_vol_%": r.std() * np.sqrt(12) * 100,
                 "min_%": r.min() * 100, "max_%": r.max() * 100, "skewness": r.skew(), "excess_kurtosis": r.kurt(),
                 "jarque_bera": jb.statistic, "jb_p": jb.pvalue, "months": len(r)})
T2 = pd.DataFrame(desc).set_index("ETF")
OUT.table(T2, "T02_descriptive_stats", f"Descriptive statistics of monthly ETF returns, {Rm.index[0]:%b %Y}-{Rm.index[-1]:%b %Y}.")
T3 = Rm.rename(columns=NAMES).corr()
OUT.table(T3, "T03_correlations", "Correlation matrix of monthly ETF returns.")
fig = S.fig_corr(Rm, NAMES); OUT.figure(fig, "F03_correlation_heatmap", "Correlation of monthly ETF returns."); plt.show()

sm = SMETA.copy()
universe = pd.DataFrame({
    "stocks ever eligible": [len(SP.columns)],
    "eligible at OOS start": [int(SE.loc[OOS[0]].sum()) if OOS[0] in SE.index else np.nan],
    "delisted / stopped before end": [int(SP.loc[OOS].iloc[-1].isna().sum())],
    "share priced from Yahoo adj. close": [float((sm["price_source"] == "yahoo_adj").mean()) if "price_source" in sm else np.nan],
    "split/bonus corrections (bhavcopy fallback)": [int(sm.get("split_corrections", pd.Series([0])).sum())],
    "monthly returns > +100% (stock-months, raw)": [int((SP_RAW.pct_change(fill_method=None) > 1.0).sum().sum())],
    "monthly returns < -60% (stock-months, raw)": [int((SP_RAW.pct_change(fill_method=None) < -0.6).sum().sum())],
    "spike-reversals fixed (|r|>50% then reversal to within 20%)": [N_SPIKES],
    "monthly returns > +100% after cleaning": [int((SP.pct_change(fill_method=None) > 1.0).sum().sum())],
    "ETF daily bad ticks fixed (price + NAV)": [int(sum(ETF_FIXED.values()))]}).T
universe.columns = ["value"]
OUT.table(universe, "T04_stock_universe", "Direct-equity universe built from NSE month-end bhavcopies (survivorship-bias free).")
display(T2.round(3)); display(universe)
""")

md("## 9. ETF tracking quality, premium/discount and liquidity")
code(r"""
te_rows = []
navw = NAVN                                 # unit-normalised NAV (same basis as split-adjusted price)
triw = S.wide_daily(etf_tri, "key", "tri") if len(etf_tri) else pd.DataFrame()
closew = S.wide_daily(etf_daily, "key", "close")   # split-adjusted, not dividend-adjusted (comparable with NAV)
volw = S.wide_daily(etf_daily, "key", "volume")
for k in KEYS:
    row = {"ETF": NAMES[k]}
    if k in triw and k in navw:
        row.update({f"nav_{a}": b for a, b in E.tracking_measures(navw[k].pct_change(), triw[k].pct_change()).items()})
    if k in triw:
        row.update(E.tracking_measures(closew[k].pct_change(), triw[k].pct_change()))
    if k in navw:
        row.update(E.premium_discount(closew[k], navw[k]))
    row.update(E.liquidity_measures(closew[k], volw[k]))
    te_rows.append(row)
T5 = pd.DataFrame(te_rows).set_index("ETF")
# show the days excluded as data errors so they can be checked against the source
bad_rows = []
for k in KEYS:
    if k not in triw:
        continue
    ri = triw[k].pct_change()
    for src, ser in (("market price", closew.get(k)), ("NAV", navw.get(k) if k in navw else None)):
        if ser is None:
            continue
        re_ = ser.pct_change()
        dd = pd.concat([re_, ri], axis=1).dropna()
        flag = dd[(dd.iloc[:, 0] - dd.iloc[:, 1]).abs() > 0.20]
        for d_, r_ in flag.iterrows():
            bad_rows.append({"ETF": NAMES[k], "series": src, "date": d_.date(), "ETF return": r_.iloc[0], "index return": r_.iloc[1],
                             "level": ser.loc[d_], "previous level": ser.shift(1).loc[d_]})
if bad_rows:
    BAD = pd.DataFrame(bad_rows)
    BAD.to_csv(f"{DIRS['tables']}/etf_data_error_days.csv", index=False)
    print("Days excluded from tracking error as data errors (|ETF - index| > 20% in one day):")
    display(BAD.round(4))
OUT.table(T5, "T05_tracking_liquidity", "Tracking error (TE1 mean absolute difference, TE2 SD of differences, TE3 regression standard error; annualised, price vs TRI; nav_ prefix = NAV vs TRI), premium/discount to NAV and liquidity.")
fig = S.fig_tracking(T5); OUT.figure(fig, "F10_tracking_premium", "Tracking error and premium/discount of the selected ETFs."); plt.show()
display(T5.T.round(4))
""")

md("## 10. Dynamic conditional correlations (DCC-GARCH)")
code(r"""
try:
    DCC = E.dcc_correlations(Rd.loc[start_common:].dropna())
    eq = KEYS[0]
    pairs = [c for c in DCC.columns if c.startswith(eq + " /")]
    DCC_l = DCC[pairs].rename(columns=lambda c: " / ".join(NAMES[x] for x in c.split(" / ")))
    fig = S.fig_dcc(DCC_l, list(DCC_l.columns))
    OUT.figure(fig, "F04_dcc_correlations", f"DCC-GARCH(1,1) correlation of the {NAMES[eq]} ETF with other ETFs (a={DCC.attrs['dcc_a']:.3f}, b={DCC.attrs['dcc_b']:.3f})."); plt.show()
    DCC_l.resample("ME").mean().to_csv(f"{DIRS['tables']}/dcc_correlations_monthly.csv")
except Exception as e:
    print("DCC skipped:", e)
""")

md("## 11. Strategy weights (out-of-sample, rolling window)")
code(r"""
BT = CONFIG["backtest"]
REBAL = E.rebalance_schedule(OOS, BT["rebalance"])
def schedules(window=BT["window"], cov=BT["main_cov"], rebal=REBAL, oos=OOS):
    out = {}
    for s in BT["strategies"]:
        out[s] = S.weight_schedule(s, cov, Rm, Rd, RF, oos, rebal, window, BT["weight_cap"], single_asset=KEYS[0],
                                   equity_assets=[k for k in KEYS if TYPES[k] == "eq_etf"])
    return out
SCHED = schedules()
dr = []
for s, w in SCHED.items():
    cov_full = E.cov_ledoit_wolf(Rm)
    dr.append({"strategy": LBL[s], "avg_diversification_ratio": np.mean([E.diversification_ratio(w.loc[d], cov_full) for d in w.index]),
               **{f"avg_w_{NAMES[k]}": w[k].mean() for k in KEYS}})
T6 = pd.DataFrame(dr).set_index("strategy")
OUT.table(T6, "T06_weights_diversification", "Average target weights and diversification ratio of ETF strategies.")
fig = S.fig_weights({LBL[s]: SCHED[s] for s in ("MinVar", "MaxSharpe", "RiskParity")}, NAMES)
OUT.figure(fig, "F09_weights", "Target weights of optimised ETF strategies at each rebalance date."); plt.show()
display(T6.round(3))
""")

md("## 12. ETF portfolio simulations (investor size × broker × tax slab × friction level)")
code(r"""
def sel(df, **kw):
    # boolean-mask filter (avoids DataFrame.query, which breaks with numexpr + string dtypes on Colab)
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]

def contrib_for(mode, amt, dates=OOS):
    return S.contributions(dates, mode, amt)

INVESTORS = [("lump", a) for a in CONFIG["investors"]["lump"]] + [("sip", a) for a in CONFIG["investors"]["sip"]]
def inv_label(m, a): return f"{'Lump' if m == 'lump' else 'SIP'} Rs {a:,.0f}" + ("/month" if m == "sip" else "")
MODES = ["frictionless", "costs", "costs_tax"]
etf_rows, ETF_RET = [], {}
t0 = time.time()
for (im, amt) in INVESTORS:
    c = contrib_for(im, amt)
    for broker in CONFIG["brokers"]:
        for mode in MODES:
            slabs = CONFIG["slabs"] if mode == "costs_tax" else [CONFIG["main_slab"]]
            if mode == "frictionless" and broker != CONFIG["main_broker"]:
                continue
            for slab in slabs:
                for s in BT["strategies"]:
                    res = S.run_one(PX_OOS, TYPES, SCHED[s], c, REBAL, mode, broker, slab, CONFIG)
                    m = E.sim_summary(res, RF_OOS)
                    etf_rows.append({"investor": inv_label(im, amt), "inv_mode": im, "amount": amt, "broker": broker,
                                     "mode": mode, "slab": slab, "strategy": s, **m})
                    ETF_RET[(im, amt, broker, mode, slab, s)] = res.returns
ETF_ALL = pd.DataFrame(etf_rows)
ETF_ALL.to_csv(f"{DIRS['tables']}/etf_all_scenarios.csv", index=False)
print(f"{len(ETF_ALL)} ETF simulations in {time.time() - t0:.0f}s")

MI, ML = CONFIG["main_investor"], CONFIG["main_lump"]
def main_key(s, mode="costs_tax", inv=MI, broker=CONFIG["main_broker"], slab=CONFIG["main_slab"]):
    return (inv["mode"], inv["amount"], broker, mode, slab if mode == "costs_tax" else CONFIG["main_slab"], s)
cols = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar95_monthly", "ceq", "annual_turnover",
        "cost_pct_contrib", "tax_pct_contrib", "tax_pct_gain", "wealth_multiple_post_tax", "xirr_post_tax"]
sub = ETF_ALL[(ETF_ALL["broker"] == CONFIG["main_broker"]) & (ETF_ALL["slab"] == CONFIG["main_slab"])
              & (((ETF_ALL["inv_mode"] == MI["mode"]) & (ETF_ALL["amount"] == MI["amount"]))
                 | ((ETF_ALL["inv_mode"] == ML["mode"]) & (ETF_ALL["amount"] == ML["amount"])))]
T7 = sub.assign(strategy=sub["strategy"].map(LBL)).set_index(["investor", "mode", "strategy"])[cols]
OUT.table(T7, "T07_etf_performance_main", f"Out-of-sample performance of ETF strategies for the main investors ({inv_label(MI['mode'], MI['amount'])}; {inv_label(ML['mode'], ML['amount'])}), {CONFIG['main_broker']} brokerage, {CONFIG['main_slab']:.0%} slab: frictionless, with costs, and with costs and tax.")
display(T7.round(4))
""")

md("## 13. Random direct-equity portfolios (survivorship-bias-free)")
code(r"""
SC = CONFIG["stocks"]
SP_OOS = SP.reindex(OOS)
GF = None
stock_rows, STOCK_RET = [], {}
t0 = time.time()
SETS = {N: S.draw_stock_sets(SP, SE, OOS[0], N, SC["draws"], SC["seed"]) for N in SC["N_list"]}
# (a) main investor, full number of draws, frictionless and costs+tax
for N, sets in SETS.items():
    for j, ids in enumerate(sets):
        for mode in ("frictionless", "costs_tax"):
            res = S.run_stock_portfolio(SP_OOS, ids, contrib_for(MI["mode"], MI["amount"]), REBAL, mode, CONFIG["main_broker"], CONFIG)
            m = E.sim_summary(res, RF_OOS)
            stock_rows.append({"N": N, "draw": j, "investor": inv_label(MI["mode"], MI["amount"]), "inv_mode": MI["mode"],
                               "amount": MI["amount"], "broker": CONFIG["main_broker"], "mode": mode, **m})
            if mode == "costs_tax":
                STOCK_RET[(N, j)] = res.returns
    print(f"  N={N}: {len(sets)} draws done ({time.time() - t0:.0f}s)")
# (b) investor-size x broker grid on a subset of draws (common random numbers)
for (im, amt) in INVESTORS:
    for broker in CONFIG["brokers"]:
        if (im, amt, broker) == (MI["mode"], MI["amount"], CONFIG["main_broker"]):
            continue
        for N, sets in SETS.items():
            for j, ids in enumerate(sets[:SC["draws_grid"]]):
                res = S.run_stock_portfolio(SP_OOS, ids, contrib_for(im, amt), REBAL, "costs_tax", broker, CONFIG)
                stock_rows.append({"N": N, "draw": j, "investor": inv_label(im, amt), "inv_mode": im, "amount": amt,
                                   "broker": broker, "mode": "costs_tax", **E.sim_summary(res, RF_OOS)})
    print(f"  grid {inv_label(im, amt)} done ({time.time() - t0:.0f}s)")
STK_ALL = pd.DataFrame(stock_rows)
STK_ALL.to_csv(f"{DIRS['tables']}/stock_all_draws.csv", index=False)
print(f"{len(STK_ALL)} stock-portfolio simulations in {time.time() - t0:.0f}s")
""")
code(r"""
main_stk = STK_ALL[(STK_ALL["investor"] == inv_label(MI["mode"], MI["amount"])) & (STK_ALL["broker"] == CONFIG["main_broker"])]
etf_main = {s: sel(ETF_ALL, inv_mode=MI["mode"], amount=MI["amount"], broker=CONFIG["main_broker"], mode="costs_tax", slab=CONFIG["main_slab"], strategy=s).iloc[0]
            for s in BT["strategies"]}
rows = []
for mode in ("frictionless", "costs_tax"):
    for N in SC["N_list"]:
        d = main_stk[(main_stk["N"] == N) & (main_stk["mode"] == mode)]
        e = sel(ETF_ALL, inv_mode=MI["mode"], amount=MI["amount"], broker=CONFIG["main_broker"], mode=mode, slab=CONFIG["main_slab"], strategy="EW").iloc[0]
        e1 = sel(ETF_ALL, inv_mode=MI["mode"], amount=MI["amount"], broker=CONFIG["main_broker"], mode=mode, slab=CONFIG["main_slab"], strategy="EQ100").iloc[0]
        e2 = sel(ETF_ALL, inv_mode=MI["mode"], amount=MI["amount"], broker=CONFIG["main_broker"], mode=mode, slab=CONFIG["main_slab"], strategy="EQ_EW").iloc[0]
        rows.append({"mode": mode, "N stocks": N, "draws": len(d),
                     "Sharpe p5": d["sharpe"].quantile(.05), "Sharpe median": d["sharpe"].median(), "Sharpe p95": d["sharpe"].quantile(.95),
                     "vol median": d["ann_vol"].median(), "MDD median": d["max_drawdown"].median(),
                     "cash drag (avg cash weight) median": d["avg_cash_weight"].median(),
                     "cost % contrib median": d["cost_pct_contrib"].median(), "tax % contrib median": d["tax_pct_contrib"].median(),
                     "1/N ETF Sharpe": e["sharpe"], "% draws below 1/N ETF Sharpe": (d["sharpe"] < e["sharpe"]).mean() * 100,
                     "% draws with higher vol than 1/N ETF": (d["ann_vol"] > e["ann_vol"]).mean() * 100,
                     "Nifty 50 ETF Sharpe": e1["sharpe"], "% draws below Nifty 50 ETF Sharpe": (d["sharpe"] < e1["sharpe"]).mean() * 100,
                     "Equity ETFs 1/N Sharpe": e2["sharpe"], "% draws below Equity ETFs 1/N Sharpe": (d["sharpe"] < e2["sharpe"]).mean() * 100,
                     "% draws with higher vol than Nifty 50 ETF": (d["ann_vol"] > e1["ann_vol"]).mean() * 100})
T8 = pd.DataFrame(rows).set_index(["mode", "N stocks"])
OUT.table(T8, "T08_stocks_vs_etf", f"Random N-stock portfolios versus ETF portfolios (equity-only: Nifty 50 ETF and Equity ETFs 1/N; multi-asset: 1/N ETF), {inv_label(MI['mode'], MI['amount'])}, {CONFIG['main_broker']} brokerage.")
display(T8.round(3))
fig = S.fig_sharpe_dist(main_stk[main_stk["mode"] == "costs_tax"][["N", "sharpe"]], {LBL[s]: etf_main[s]["sharpe"] for s in ("EQ100", "EQ_EW", "EW")})
OUT.figure(fig, "F05_sharpe_distribution", "Distribution of net Sharpe ratios of random N-stock portfolios against ETF strategies (SIP investor, costs and tax)."); plt.show()
N_band = 10 if 10 in SC["N_list"] else SC["N_list"][0]
band = pd.DataFrame({j: STOCK_RET[(N_band, j)] for j in range(len(SETS[N_band]))})
ret_main = {LBL[s]: ETF_RET[main_key(s)] for s in BT["strategies"]}
fig = S.fig_cumulative(ret_main, band, f"Random {N_band}-stock")
OUT.figure(fig, "F01_cumulative_growth", "Growth of Rs 1 (time-weighted, net of costs and tax): ETF strategies vs random 10-stock portfolios."); plt.show()
""")

md("### 13b. Like-for-like check and risk decomposition\nThe main universe (500 most-traded stocks) includes mid and small caps. Here stocks are also drawn from the 100 most-traded stocks only. Ranking is by traded value (free data has no market cap), so this universe is liquid but still includes high-beta, heavily traded names.\n\nTable T16 separates two effects: **systematic** (an equal-weighted basket of the whole universe vs the cap-weighted Nifty 50 ETF) and **diversifiable** (a small random N-stock portfolio vs that whole-universe basket). The universe baskets are frictionless and not investable by a small investor; they are reference points only.")
code(r"""
T15 = None
if SRANK is not None:
    LC = SC["large_cap_top_n"]
    ELIG_LC = (SRANK <= LC).reindex(columns=SP.columns).fillna(False)
    rows, LC_RET = [], {}
    for N in SC["N_list"]:
        sets_lc = S.draw_stock_sets(SP, ELIG_LC, OOS[0], N, SC["draws"], SC["seed"] + 7)
        for j, ids in enumerate(sets_lc):
            res = S.run_stock_portfolio(SP_OOS, ids, contrib_for(MI["mode"], MI["amount"]), REBAL, "costs_tax", CONFIG["main_broker"], CONFIG)
            rows.append({"N": N, "draw": j, **E.sim_summary(res, RF_OOS)})
            LC_RET[(N, j)] = res.returns
    LCD = pd.DataFrame(rows)
    LCD.to_csv(f"{DIRS['tables']}/stock_largecap_draws.csv", index=False)
    e1, e2 = etf_main["EQ100"], etf_main["EQ_EW"]
    out = []
    for N in SC["N_list"]:
        d = LCD[LCD["N"] == N]
        out.append({"N stocks": N, "draws": len(d), "Sharpe p5": d["sharpe"].quantile(.05), "Sharpe median": d["sharpe"].median(),
                    "Sharpe p95": d["sharpe"].quantile(.95), "vol median": d["ann_vol"].median(), "MDD median": d["max_drawdown"].median(),
                    "XIRR median": d["xirr_post_tax"].median(),
                    "% draws below Nifty 50 ETF Sharpe": (d["sharpe"] < e1["sharpe"]).mean() * 100,
                    "% draws below Equity ETFs 1/N Sharpe": (d["sharpe"] < e2["sharpe"]).mean() * 100,
                    "% draws with higher vol than Nifty 50 ETF": (d["ann_vol"] > e1["ann_vol"]).mean() * 100,
                    "% draws with deeper MDD than Nifty 50 ETF": (d["max_drawdown"] < e1["max_drawdown"]).mean() * 100})
    T15 = pd.DataFrame(out).set_index("N stocks")
    OUT.table(T15, "T15_largecap_universe", f"Random N-stock portfolios drawn from the {LC} most-traded stocks (large-cap, Nifty 100-like universe) versus equity ETFs, {inv_label(MI['mode'], MI['amount'])}, net of costs and tax. Nifty 50 ETF Sharpe {e1['sharpe']:.2f}, vol {e1['ann_vol']:.1%}, MDD {e1['max_drawdown']:.1%}.")
    display(T15.round(3))

    # ---- T16: universe equal-weight baskets and risk decomposition ----
    def beta_vs(r, m):
        a, b = r.align(m, join="inner")
        return float(np.cov(a, b)[0, 1] / np.var(b, ddof=1))
    mret = ETF_RET[main_key("EQ100")]
    dec = []
    def dec_row(name, r=None, m=None):
        if m is None:
            m = E.perf_metrics(r, RF_OOS)
        dec.append({"portfolio": name, "ann_return": m["ann_return"], "ann_vol": m["ann_vol"], "sharpe": m["sharpe"],
                    "max_drawdown": m["max_drawdown"], "beta vs Nifty 50 ETF": beta_vs(r, mret) if r is not None else np.nan})
    dec_row("Nifty 50 ETF (cap-weighted)", mret)
    UNIV_RET = {}
    for lbl, elig in ((f"top-{LC}", ELIG_LC), (f"top-{SC['top_n']}", SE)):
        ids_u = [c for c in SP.columns if bool(elig.loc[OOS[0], c]) and np.isfinite(SP.loc[OOS[0], c])]
        res_u = S.run_stock_portfolio(SP_OOS, ids_u, contrib_for(MI["mode"], MI["amount"]), REBAL, "frictionless", CONFIG["main_broker"], CONFIG)
        UNIV_RET[lbl] = res_u.returns
        dec_row(f"All {len(ids_u)} {lbl} stocks, equal-weight (frictionless reference)", res_u.returns)
    for lbl, src, store in ((f"top-{LC}", LCD, LC_RET), (f"top-{SC['top_n']}", main_stk[main_stk["mode"] == "costs_tax"], STOCK_RET)):
        for N in SC["N_list"]:
            keys = [k for k in store if k[0] == N]
            betas = [beta_vs(store[k], mret) for k in keys]
            d = src[src["N"] == N]
            dec.append({"portfolio": f"Random {N} stocks from {lbl} (median of draws)", "ann_return": d["ann_return"].median(),
                        "ann_vol": d["ann_vol"].median(), "sharpe": d["sharpe"].median(), "max_drawdown": d["max_drawdown"].median(),
                        "beta vs Nifty 50 ETF": float(np.median(betas))})
    T16 = pd.DataFrame(dec).set_index("portfolio")
    OUT.table(T16, "T16_risk_decomposition", f"Risk decomposition, {inv_label(MI['mode'], MI['amount'])}: cap-weighted Nifty 50 ETF vs equal-weighted whole-universe baskets (systematic difference) vs random N-stock portfolios (adds diversifiable risk). Random portfolios net of costs and tax; universe baskets frictionless.")
    display(T16.round(3))
else:
    print("stock_rank.csv.gz not found - re-run Section 6d (uses cached downloads) to enable this check.")
""")

md("## 14. Statistical tests\nLedoit–Wolf (2008) Sharpe test, Ledoit–Wolf (2011) variance test, Romano–Wolf (2005) stepdown.")
code(r"""
TS = CONFIG["tests"]
ex = {s: (ETF_RET[main_key(s)] - RF_OOS.reindex(ETF_RET[main_key(s)].index)).dropna() for s in BT["strategies"]}
rows = []
for bench in ("EQ100", "EW"):
    for s in BT["strategies"]:
        if s == bench: continue
        a, b = ex[s].align(ex[bench], join="inner")
        sh = E.ledoit_wolf_sharpe_test(a.values, b.values, TS["block"], TS["bootstrap_M"], TS["seed"])
        va = E.ledoit_wolf_variance_test(a.values, b.values, TS["block"], TS["bootstrap_M"], TS["seed"])
        rows.append({"benchmark": LBL[bench], "strategy": LBL[s], "Sharpe diff (ann.)": sh["diff_annualised"],
                     "Sharpe p (LW2008)": sh["p_value"], "log variance ratio": va["diff"], "variance p (LW2011)": va["p_value"]})
T9 = pd.DataFrame(rows).set_index(["benchmark", "strategy"])
rw = E.romano_wolf_sharpe(ex["EW"].values, {LBL[s]: ex[s].values for s in BT["strategies"] if s != "EW"},
                          TS["block"], TS["bootstrap_M"], TS["seed"]).set_index("strategy")
T9 = T9.join(rw["rw_adj_p"].rename("Romano-Wolf adj. p (vs 1/N)"), on="strategy")
T9.loc[T9.index.get_level_values("benchmark") != LBL["EW"], "Romano-Wolf adj. p (vs 1/N)"] = np.nan
OUT.table(T9, "T09_tests_etf", f"Tests of ETF strategies against the Nifty 50 ETF and the 1/N ETF portfolio (net of costs and tax; circular block bootstrap, b={TS['block']}, M={TS['bootstrap_M']}).")
display(T9.round(4))

# ETF portfolios against every random stock draw (equity-only and multi-asset benchmarks)
rows = []
for bench in ("EQ100", "EQ_EW", "EW"):
    for N in SC["N_list"]:
        pv, diffs = [], []
        for j in range(len(SETS[N])):
            r = STOCK_RET[(N, j)]
            se = (r - RF_OOS.reindex(r.index)).dropna()
            a, b = ex[bench].align(se, join="inner")
            if len(a) < 24 or b.std() == 0: continue
            t = E.ledoit_wolf_sharpe_test(a.values, b.values, TS["block"], TS["stock_test_M"], TS["seed"])
            pv.append(t["p_value"]); diffs.append(t["diff_annualised"])
        pv, diffs = np.array(pv), np.array(diffs)
        rows.append({"ETF benchmark": LBL[bench], "N stocks": N, "draws tested": len(pv),
                     "median Sharpe diff (ETF - stocks)": np.median(diffs),
                     "% ETF significantly better (p<0.05)": np.mean((pv < TS["alpha"]) & (diffs > 0)) * 100,
                     "% stocks significantly better (p<0.05)": np.mean((pv < TS["alpha"]) & (diffs < 0)) * 100,
                     "% not significant": np.mean(pv >= TS["alpha"]) * 100})
T10 = pd.DataFrame(rows).set_index(["ETF benchmark", "N stocks"])
OUT.table(T10, "T10_tests_etf_vs_stocks", "Ledoit-Wolf Sharpe tests of ETF portfolios against each random N-stock portfolio (net of costs and tax).")
display(T10.round(2))
""")

md("## 15. Cost and tax erosion by investor size, broker and tax slab")
code(r"""
rows = []
for (im, amt) in INVESTORS:
    for broker in CONFIG["brokers"]:
        e = sel(ETF_ALL, inv_mode=im, amount=amt, broker=broker, mode="costs_tax", slab=CONFIG["main_slab"], strategy="EW")
        st = sel(STK_ALL, inv_mode=im, amount=amt, broker=broker, mode="costs_tax", N=N_band)
        if e.empty or st.empty: continue
        e = e.iloc[0]
        rows.append({"investor": inv_label(im, amt), "broker": broker,
                     "1/N ETF Sharpe": e["sharpe"], f"{N_band}-stock median Sharpe": st["sharpe"].median(),
                     "1/N ETF cost %": e["cost_pct_contrib"] * 100, f"{N_band}-stock cost % (median)": st["cost_pct_contrib"].median() * 100,
                     "1/N ETF tax %": e["tax_pct_contrib"] * 100, f"{N_band}-stock tax % (median)": st["tax_pct_contrib"].median() * 100,
                     "1/N ETF tax % of gains": e["tax_pct_gain"] * 100, f"{N_band}-stock tax % of gains (median)": st["tax_pct_gain"].median() * 100,
                     "1/N ETF XIRR post-tax": e["xirr_post_tax"], f"{N_band}-stock XIRR median": st["xirr_post_tax"].median(),
                     "1/N ETF cash weight": e["avg_cash_weight"], f"{N_band}-stock cash weight (median)": st["avg_cash_weight"].median()})
T11 = pd.DataFrame(rows).set_index(["investor", "broker"])
OUT.table(T11, "T11_erosion_by_investor", f"Costs and taxes as % of contributions, cash drag and net performance by investor size and broker: 1/N ETF vs median random {N_band}-stock portfolio ({CONFIG['main_slab']:.0%} slab).")
display(T11.round(3))
er = []
for (inv, broker), r in T11.iterrows():
    if broker != "flat": continue
    er.append({"investor": inv, "portfolio": "1/N ETF", "cost_pct": r["1/N ETF cost %"], "tax_pct": r["1/N ETF tax %"]})
    er.append({"investor": inv, "portfolio": f"{N_band} stocks", "cost_pct": r[f"{N_band}-stock cost % (median)"], "tax_pct": r[f"{N_band}-stock tax % (median)"]})
if er:
    fig = S.fig_erosion(pd.DataFrame(er))
    OUT.figure(fig, "F06_cost_tax_erosion", f"Costs (solid) and taxes (hatched) as % of contributions with Rs 20 flat brokerage: 1/N ETF vs median {N_band}-stock portfolio."); plt.show()

T12 = (ETF_ALL[(ETF_ALL["mode"] == "costs_tax") & (ETF_ALL["broker"] == CONFIG["main_broker"])
              & (ETF_ALL["inv_mode"] == MI["mode"]) & (ETF_ALL["amount"] == MI["amount"])]
       .assign(strategy=lambda d: d["strategy"].map(LBL))
       .pivot_table(index="strategy", columns="slab", values=["sharpe", "xirr_post_tax", "tax_pct_contrib"]))
T12.columns = [f"{a} | slab {b:.0%}" for a, b in T12.columns]
OUT.table(T12, "T12_slab_scenarios", "ETF strategies under 5%, 20% and 30% income-tax slabs (slab affects debt ETF gains and short-term gold/international gains).")
display(T12.round(4))
""")

md("## 16. Sub-period analysis")
code(r"""
sp = {k: v for k, v in CONFIG["subperiods"].items()}
ret_sp = dict(ret_main)
ret_sp[f"Random {N_band}-stock (cross-sectional median monthly return)"] = band.median(axis=1)
T13 = S.subperiod_table(ret_sp, RF_OOS, sp)
OUT.table(T13.set_index(["subperiod", "strategy"]), "T13_subperiods", "Performance by sub-period (main SIP investor, net of costs and tax).")
display(T13.round(3))
""")

md("## 17. Robustness checks")
code(r"""
rows = []
c_main = contrib_for(MI["mode"], MI["amount"])
def add(check, variant, s, res):
    m = E.sim_summary(res, RF_OOS.reindex(res.returns.index))
    rows.append({"check": check, "variant": variant, "strategy": LBL[s], "sharpe": m["sharpe"], "ann_return": m["ann_return"],
                 "max_drawdown": m["max_drawdown"], "xirr_post_tax": m["xirr_post_tax"], "cost_pct_contrib": m["cost_pct_contrib"]})
base_args = dict(mode="costs_tax", broker=CONFIG["main_broker"], slab=CONFIG["main_slab"], cfg=CONFIG)
ROB = CONFIG["robustness"]
# (1) covariance estimator
for cov in BT["cov_methods"]:
    sch = SCHED if cov == BT["main_cov"] else schedules(cov=cov)
    for s in ("MinVar", "MaxSharpe", "RiskParity"):
        add("covariance", cov, s, S.run_one(PX_OOS, TYPES, sch[s], c_main, REBAL, **base_args))
# (2) estimation window (common OOS start = longest window, for comparability)
wmax = max(ROB["windows"])
oos_w = PXm.loc[start_common:].index[wmax:]
for w in ROB["windows"]:
    rb = E.rebalance_schedule(oos_w, BT["rebalance"])
    sch = schedules(window=w, rebal=rb, oos=oos_w)
    for s in ("EW", "MinVar", "MaxSharpe"):
        add("window (common start)", f"{w} months", s, S.run_one(PXm.loc[oos_w], TYPES, sch[s], contrib_for(MI["mode"], MI["amount"], oos_w), rb, **base_args))
# (3) rebalancing frequency
for fq in ROB["rebalance"]:
    rb = set() if fq == "T5" else E.rebalance_schedule(OOS, fq)
    sch = schedules(rebal=E.rebalance_schedule(OOS, "A") if fq == "T5" else rb)
    for s in ("EW", "MinVar"):
        add("rebalancing", "5% threshold" if fq == "T5" else fq, s,
            S.run_one(PX_OOS, TYPES, sch[s], c_main, rb, threshold=0.05 if fq == "T5" else None, **base_args))
# (4) extra transaction costs
for bps in ROB["extra_bps"]:
    for s in ("EQ100", "EW", "MinVar"):
        add("extra cost", f"+{bps} bps", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, extra_bps=bps, **base_args))
# (5) historical tax rules with grandfathering
gf = PXm.loc[:"2018-01-31"].iloc[-1].to_dict() if PXm.index[0] <= pd.Timestamp("2018-01-31") else None
for s in ("EQ100", "EW", "MinVar"):
    add("tax regime", "historical (date-dependent)", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, regime="historical", gf_prices=gf, **base_args))
    add("tax regime", "current FY2026-27", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, **base_args))
# (6) flat brokerage
for s in ("EQ100", "EW", "MinVar"):
    add("brokerage", "Rs 20 flat per order", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, mode="costs_tax", broker="flat", slab=CONFIG["main_slab"], cfg=CONFIG))
# (6b) SIP order rule: one order per month (most underweight ETF) vs split across all ETFs, zero brokerage
import copy
cfg_single = copy.deepcopy(CONFIG); cfg_single["sip_single_order_brokers"] = ["zero", "flat"]
for s in ("EW", "MinVar", "RiskParity"):
    add("SIP order rule", "split across all holdings (main, zero brokerage)", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, **base_args))
    add("SIP order rule", "one order per month (zero brokerage)", s,
        S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, mode="costs_tax", broker=CONFIG["main_broker"], slab=CONFIG["main_slab"], cfg=cfg_single))
# (6c) cleaned vs raw stock prices: median Sharpe of the same random 10-stock draws
if CONFIG.get("clean_spikes") and N_SPIKES:
    SPR_OOS = SP_RAW.reindex(OOS)
    for lbl_, panel in (("cleaned (main)", SP_OOS), ("raw, uncleaned", SPR_OOS)):
        sh = [E.sim_summary(S.run_stock_portfolio(panel, ids, c_main, REBAL, "costs_tax", CONFIG["main_broker"], CONFIG), RF_OOS)
              for ids in SETS[N_band][:SC["draws_grid"]]]
        rows.append({"check": "stock price cleaning", "variant": lbl_, "strategy": f"Random {N_band} stocks (median)",
                     "sharpe": np.median([m["sharpe"] for m in sh]), "ann_return": np.median([m["ann_return"] for m in sh]),
                     "max_drawdown": np.median([m["max_drawdown"] for m in sh]), "xirr_post_tax": np.median([m["xirr_post_tax"] for m in sh]),
                     "cost_pct_contrib": np.median([m["cost_pct_contrib"] for m in sh])})
# (7) debt sleeve priced at market price instead of NAV
if CONFIG.get("price_source"):
    PXm_mkt = PXd_MKT.resample("ME").last()
    for s in ("EW", "MinVar", "RiskParity"):
        add("debt pricing", "market price (thin trading)", s, S.run_one(PXm_mkt.loc[OOS], TYPES, SCHED[s], c_main, REBAL, **base_args))
        add("debt pricing", "NAV (main)", s, S.run_one(PX_OOS, TYPES, SCHED[s], c_main, REBAL, **base_args))
T14 = pd.DataFrame(rows).set_index(["check", "variant", "strategy"])
OUT.table(T14, "T14_robustness", "Robustness of ETF results (main SIP investor, net of costs and tax).")
display(T14.round(4))
""")

md("## 18. Additional figures")
code(r"""
fig = S.fig_drawdowns(ret_main); OUT.figure(fig, "F02_drawdowns", "Drawdowns of ETF strategies (net of costs and tax)."); plt.show()
fig = S.fig_rolling_sharpe(ret_main, RF_OOS); OUT.figure(fig, "F08_rolling_sharpe", "Rolling 36-month Sharpe ratio of ETF strategies."); plt.show()
cloud = main_stk[(main_stk["mode"] == "frictionless") & (main_stk["N"] == N_band)]
pts = {LBL[s]: SCHED[s].mean() for s in ("EW", "MinVar", "MaxSharpe")}
fig = S.fig_frontier(Rm, NAMES, pts, cloud)
OUT.figure(fig, "F07_efficient_frontier", "Long-only ETF efficient frontier (full sample, Ledoit-Wolf covariance) with average strategy weights and realised random 10-stock portfolios."); plt.show()
""")

md("## 19. Export, key results and run log")
code(r"""
wb = OUT.write_workbook()
e = {s: etf_main[s] for s in BT["strategies"]}
t8 = T8.loc["costs_tax"]
lines = [f"# Key results - run {RUN_ID}", "",
         f"Data mode: **{DATA_MODE}**" + (" (SMOKE TEST ONLY - NOT FOR THE PAPER)" if DATA_MODE == "synthetic" else ""),
         f"Out-of-sample period: {OOS[0]:%b %Y} to {OOS[-1]:%b %Y} ({len(OOS)} months). ETFs: {', '.join(NAMES.values())}.",
         f"Main investor: {inv_label(MI['mode'], MI['amount'])}, {CONFIG['main_broker']} brokerage, {CONFIG['main_slab']:.0%} slab, current tax rules.", "",
         "## ETF strategies (net of costs and tax)"]
for s in BT["strategies"]:
    lines.append(f"- {LBL[s]}: Sharpe {e[s]['sharpe']:.2f}, annual return {e[s]['ann_return']:.1%}, volatility {e[s]['ann_vol']:.1%}, max drawdown {e[s]['max_drawdown']:.1%}, post-tax XIRR {e[s]['xirr_post_tax']:.1%}.")
lines += ["", "## Random direct-equity portfolios vs 1/N ETF"]
for N, r in t8.iterrows():
    lines.append(f"- {N} stocks: median Sharpe {r['Sharpe median']:.2f} (5th-95th pct {r['Sharpe p5']:.2f} to {r['Sharpe p95']:.2f}); "
                 f"{r['% draws below 1/N ETF Sharpe']:.0f}% of draws had a lower Sharpe than the 1/N ETF; "
                 f"1/N ETF significantly better in {T10.loc[(LBL['EW'], N), '% ETF significantly better (p<0.05)']:.0f}% of LW tests; "
                 f"{r['% draws below Nifty 50 ETF Sharpe']:.0f}% of draws below the Nifty 50 ETF and {r['% draws below Equity ETFs 1/N Sharpe']:.0f}% below the equity-ETF 1/N.")
lines += ["", "## Equity-only comparison (like-for-like)"]
for N in SC["N_list"]:
    r10 = T10.loc[(LBL["EQ100"], N)]
    lines.append(f"- {N} stocks (top-500 universe) vs Nifty 50 ETF: median Sharpe difference (ETF - stocks) {r10['median Sharpe diff (ETF - stocks)']:+.2f}; "
                 f"ETF significantly better in {r10['% ETF significantly better (p<0.05)']:.0f}%, stocks in {r10['% stocks significantly better (p<0.05)']:.0f}% of tests.")
if T15 is not None:
    for N, r in T15.iterrows():
        lines.append(f"- {N} stocks (large-cap universe): median Sharpe {r['Sharpe median']:.2f}, {r['% draws below Nifty 50 ETF Sharpe']:.0f}% of draws below the Nifty 50 ETF, "
                     f"{r['% draws with higher vol than Nifty 50 ETF']:.0f}% with higher volatility, {r['% draws with deeper MDD than Nifty 50 ETF']:.0f}% with a deeper drawdown.")
if "T16" in globals():
    lines += ["", "## Risk decomposition (T16)"]
    for nm, r in T16.iterrows():
        lines.append(f"- {nm}: Sharpe {r['sharpe']:.2f}, volatility {r['ann_vol']:.1%}, max drawdown {r['max_drawdown']:.1%}, beta {r['beta vs Nifty 50 ETF']:.2f}.")
lines += ["", "## Optimised vs 1/N (Romano-Wolf adjusted p-values)"]
for st, r in T9.xs(LBL["EW"], level="benchmark").iterrows():
    lines.append(f"- {st}: Sharpe difference {r['Sharpe diff (ann.)']:+.2f}, LW p = {r['Sharpe p (LW2008)']:.3f}, RW adj. p = {r['Romano-Wolf adj. p (vs 1/N)']:.3f}.")
lines += ["", "## Files", f"- Tables: {DIRS['tables']} (CSV, LaTeX, {os.path.basename(wb)})", f"- Figures: {DIRS['figures']} (PNG 300 dpi, PDF)"]
txt = "\n".join(lines)
open(f"{RUN_DIR}/key_results.md", "w").write(txt)
open(f"{DIRS['manuscript']}/key_results_{RUN_ID}.md", "w").write(txt)

import importlib.metadata as md_
vers = {p: md_.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "arch", "yfinance", "requests", "openpyxl"]
        if importlib.util.find_spec(p.replace("scikit-learn", "sklearn")) is not None}
log = {"run_id": RUN_ID, "data_mode": DATA_MODE, "quick_run": QUICK_RUN, "python": platform.python_version(),
       "platform": platform.platform(), "packages": vers, "runtime_minutes": round((time.time() - T0) / 60, 1),
       "oos_start": str(OOS[0].date()), "oos_end": str(OOS[-1].date()), "n_months": len(OOS),
       "etfs": SEL.to_dict(orient="records"), "n_etf_sims": len(ETF_ALL), "n_stock_sims": len(STK_ALL)}
json.dump(log, open(f"{DIRS['logs']}/run_log.json", "w"), indent=2, default=str)
if os.path.exists(MAN.path):
    MAN.table().to_csv(f"{DIRS['logs']}/download_manifest_snapshot.csv", index=False)
import shutil
zip_path = shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
print(txt)
print("\nAll outputs:", RUN_DIR, "\nZip:", zip_path)
display(pd.DataFrame(OUT.index))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
out = os.path.join(HERE, "notebooks", "ETF_SmallInvestor_India.ipynb")
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
