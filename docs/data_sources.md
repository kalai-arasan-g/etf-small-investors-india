# Data sources (all free and public)

| Data | Source | Access in code | Used for |
|---|---|---|---|
| ETF daily prices and volumes (close, split-adjusted close, adjusted close) | Yahoo Finance | `yfinance` (`data_io.yf_history`) | ETF returns, liquidity, premium/discount |
| ETF daily NAV | AMFI NAV history via https://api.mfapi.in | `data_io.mfapi_nav` | Tracking error, premium/discount, G-Sec returns |
| Risk-free rate | Nippon India Liquid Fund (Direct, Growth) NAV via mfapi.in | `data_io.mfapi_nav` | Excess returns, Sharpe ratios |
| NIFTY 50 and NIFTY NEXT 50 Total Return Index | https://www.niftyindices.com (Reports → Historical Data → Total Returns Index) | manual yearly CSVs, read by `data_io.read_tri_files` | Tracking error benchmark |
| All listed equities, month-end | NSE capital-market bhavcopy archives (old format to July 2024, UDiFF format after) | `data_io.month_end_bhavcopies` | Survivorship-bias-free stock universe |
| Stock adjusted prices | Yahoo Finance; bhavcopy closes with split/bonus correction as fallback | `data_io.build_stock_panel` | Random stock portfolios |

The notebook also tries the FRED 3-month Treasury-bill series for India (INTGSTINM193N); in the published run it was unavailable, so the liquid-fund return was used for the whole period.

Selected ETFs: NIFTYBEES (Nifty 50), JUNIORBEES (Nifty Next 50), GOLDBEES (gold), LICNETFGSC (G-Sec 8–13 yr), MON100 (Nasdaq 100). Selection rule: among NSE-listed candidates for each asset class, keep those above a minimum median daily traded value over the last two years, then take the one with the longest history.

Raw data are not redistributed. Every file downloaded for the published run is listed, with URL, UTC time, row count and SHA-256 checksum, in `results/run_20261026_0831/logs/download_manifest_snapshot.csv`.
