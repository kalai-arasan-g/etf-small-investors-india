# Transaction-cost and tax rules used

## Transaction costs (delivery trades on the NSE)

| Item | Rate | Note |
|---|---|---|
| Brokerage | Rs 0 (zero-brokerage account) or Rs 20 per order (flat-fee account) | Both reported |
| STT, stocks | 0.1% on buy and on sell | Statutory |
| STT, equity ETFs | 0.001% on sell only | Statutory |
| STT, gold / G-Sec / international ETFs | Nil | Not equity-oriented |
| Exchange transaction charge | 0.00297% | Typical NSE rate |
| SEBI turnover fee | Rs 10 per crore | Statutory |
| Stamp duty | 0.015% on buy value | Statutory (since 1 July 2020) |
| GST | 18% on brokerage, exchange and SEBI fees | Statutory |
| DP charge | Rs 15.93 per security sold (incl. GST) | Typical discount broker |
| Half bid-ask spread | Equity ETFs 0.05%; stocks and gold 0.10%; G-Sec 0.15%; Nasdaq 100 ETF 0.20% | Assumption; robustness adds 10/25/50 bp |

Source for statutory levies: NSE, "Turnover fees, STT and other levies".
With flat brokerage, a SIP instalment is placed as one order in the most underweight holding.

## Capital-gains tax, main analysis (FY 2026-27 rules applied to all sales)

| Asset | Long-term after | Short-term | Long-term | Exemption |
|---|---|---|---|---|
| Stocks, equity ETFs | 12 months | 20% | 12.5% | Rs 1.25 lakh per year on long-term equity gains |
| Gold ETF | 12 months | slab | 12.5% | none |
| International ETF | 12 months | slab | 12.5% | none |
| Debt (G-Sec) ETF bought after 1 April 2023 | n.a. | slab | slab | none |

Plus 4% cess; no surcharge; FIFO lots; buying costs added to cost and selling costs deducted from sale value (except STT); short-term losses set off against any gain, long-term losses only against long-term gains, unused losses carried forward (8-year limit not modelled); tax paid each July; all holdings sold and taxed at the end. Slab rate 20% (5% and 30% as scenarios).

## Historical regime (robustness)

Rules in force on each sale date: equity LTCG exempt before 1 April 2018; 10% above Rs 1 lakh with 31 January 2018 grandfathering until 22 July 2024; equity STCG 15% until 22 July 2024; non-equity units sold before 23 July 2024 long-term after 36 months at 20% with CII indexation; Section 50AA slab taxation for debt units bought from 1 April 2023 and for gold/international units bought from 1 April 2023 and sold before 1 April 2025. Full table: Appendix B of the paper; CII values in `logs/config_used.json`.
