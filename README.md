# Do Index ETFs Help Small Investors Diversify?

**Evidence from India with Realistic Costs, Taxes and Whole-Unit Purchases**

Kalaiarasan G<sup>1</sup> and A. V. Shyam<sup>2</sup>
<sup>1</sup> Amrita School of Artificial Intelligence, <sup>2</sup> Amrita School of Business, Amrita Vishwa Vidyapeetham, Coimbatore, India

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kalai-arasan-g/etf-small-investors-india/blob/main/notebooks/ETF_SmallInvestor_India.ipynb)

Replication code and results for the paper. The code simulates a small Indian investor who buys **whole units**, invests by **lump sum or monthly SIP**, and pays **Indian transaction costs and capital-gains tax**, and compares six ETF strategies with 4,000 random stock portfolios drawn from a **survivorship-bias-free NSE universe** (January 2018 to September 2026). Only **free public data** are used.

## Main results (SIP investor, Rs 5,000 per month, net of costs and tax)

| Portfolio | Sharpe | Volatility | Max drawdown |
|---|---|---|---|
| Nifty 50 ETF | 0.30 | 16.6% | -28.8% |
| Random 10-stock portfolios (median of 500) | 0.34 | 30.0% | -56.9% |
| Multi-asset 1/N ETF portfolio (equity, gold, G-Sec, Nasdaq 100) | 0.82 | 10.4% | -10.7% |

- For Indian equity alone, the ETF gives about the same Sharpe ratio as small stock portfolios but much lower risk.
- About two-thirds of the extra risk of stock portfolios comes from equal weighting of high-beta stocks, not from holding too few stocks.
- Simple equal weights across ETFs are as good as optimised weights; fixed brokerage fees and the tax treatment of non-equity ETFs matter most for small investors.

Full numbers: [`results/run_20261026_0831/key_results.md`](results/run_20261026_0831/key_results.md).

## Repository structure

```
notebooks/ETF_SmallInvestor_India.ipynb   Google Colab notebook: downloads data, runs everything, saves all outputs
src/etf_engine.py      cost model, FIFO capital-gains tax engine, small-investor simulator,
                       portfolio rules, performance measures, Ledoit-Wolf and Romano-Wolf tests
src/data_io.py         free-data download (NSE bhavcopy, Yahoo Finance, AMFI NAV, NIFTY TRI),
                       survivorship-free stock panel, data cleaning, download manifest
src/study.py           strategy schedules, scenario runs, random stock portfolios, tables and figures
scripts/build_notebook.py     rebuilds the notebook from src/ (the notebook embeds the modules)
scripts/make_latex_tables.py  converts result CSVs into the LaTeX tables of the paper
tests/                 unit tests (pytest)
results/run_20261026_0831/    results reported in the paper: tables (CSV, LaTeX, Excel), figures (PNG, PDF),
                              configuration, run log and download manifest with SHA-256 checksums
docs/                  data sources, cost and tax rules, assumptions, references (BibTeX)
```

## How to reproduce

**Google Colab (recommended).** Open the notebook with the badge above and choose *Runtime → Run all*.
1. Allow Google Drive access. Everything is saved to `MyDrive/ETF_SmallInvestor_India/`.
2. Set `QUICK_RUN = True` for a fast check (about 20 minutes on the first run), then `QUICK_RUN = False` for the paper's numbers (500 draws per portfolio size, 4,999 bootstrap replications; about 55 minutes once data are cached).
3. Optional: NIFTY 50 and NIFTY NEXT 50 Total Return Index files must be downloaded manually from niftyindices.com (Reports → Historical Data → Total Returns Index), one CSV per year, saved as `data/raw/index_tri/NIFTY_50_<year>.csv` and `NIFTY_NEXT_50_<year>.csv`. They are needed only for the tracking-error table.
4. `DATA_MODE = "synthetic"` runs the whole pipeline in a few minutes on simulated data. Use it only to test the code.

**Locally.**
```bash
pip install -r requirements.txt
pytest -q                                   # unit tests
python scripts/build_notebook.py            # only if you change src/
jupyter notebook notebooks/ETF_SmallInvestor_India.ipynb
```

## Data

All inputs are free and public: NSE capital-market bhavcopy archives, Yahoo Finance (via `yfinance`), AMFI NAV history (via mfapi.in) and NSE Indices total return indices. Raw data are **not** included in this repository because of the terms of use of these sources; the notebook downloads them and logs every file (URL, time, rows, SHA-256) in `data/download_manifest.jsonl`. The manifest of the published run is in `results/run_20261026_0831/logs/`. See [`docs/data_sources.md`](docs/data_sources.md).

Free data have known limitations (stocks are ranked by traded value because shares outstanding are not available; Yahoo coverage of delisted stocks is incomplete). These, and the cleaning rules used, are documented in [`docs/methodology_and_assumptions.md`](docs/methodology_and_assumptions.md) and in the paper.

## Mapping of outputs to the paper

| Paper | Output file (results/run_20261026_0831/tables) |
|---|---|
| Table 1 | T01_etf_selection, T02_descriptive_stats |
| Table 2 | T05_tracking_liquidity |
| Tables 3–4 | cost and tax parameters in `logs/config_used.json` (see docs/tax_cost_rules.md) |
| Table 5 | T07_etf_performance_main |
| Table 6 | T09_tests_etf |
| Table 7 | T08_stocks_vs_etf, T10_tests_etf_vs_stocks |
| Table 8 | T16_risk_decomposition |
| Table 9 | T11_erosion_by_investor |
| Tables A1–A6 | T03, T06, T12, T13, T15, T14 |
| Figures 1–4 | F04, F05, F01, F06; Figures A1–A4: F07, F09, F02, F08 |

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). Citation details will be updated on publication.

## Use of AI tools

Claude (Anthropic) was used to assist with code development and drafting. The authors reviewed, verified and edited all code, results and text and take full responsibility for them.

## Disclaimer

This is academic research, not investment advice. Past performance does not guarantee future results.

## License

Code: MIT License (see [`LICENSE`](LICENSE)). Results and documentation: CC BY 4.0.
