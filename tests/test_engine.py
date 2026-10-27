"""Unit tests for the cost model, tax engine, simulator, statistical tests and data cleaning.

Run from the repository root:  pytest -q
"""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import data_io as D
import etf_engine as E


# ----------------------------------------------------------------------------- tax
def test_current_regime_set_off_and_exemption():
    """Short-term loss offsets the slab-taxed gain first; equity LTCG gets the Rs 1.25 lakh exemption."""
    T = E.TaxEngine(regime="current", slab=0.30)
    d0, d1 = pd.Timestamp("2024-01-31"), pd.Timestamp("2025-06-30")
    T.record("A", "eq_etf", d0, d1, 300_000, 100_000, 1)                     # LT equity gain 2,00,000
    T.record("G", "gold_etf", d0, d1, 150_000, 100_000, 1)                   # LT gold gain 50,000
    T.record("D", "debt_etf", d0, d1, 120_000, 100_000, 1)                   # debt gain 20,000 (slab)
    T.record("S", "stock", pd.Timestamp("2025-01-31"), d1, 80_000, 100_000, 1)  # ST loss 20,000
    expected = (75_000 * 0.125 + 50_000 * 0.125) * 1.04
    assert T.settle(2025) == pytest.approx(expected)


def test_historical_regime_grandfathering_and_indexation():
    H = E.TaxEngine(regime="historical", slab=0.30, gf_prices={"A": 150.0})
    # equity bought 2016, sold 2019: cost stepped up to 31-Jan-2018 FMV, gain within Rs 1 lakh exemption
    H.record("A", "eq_etf", pd.Timestamp("2016-01-31"), pd.Timestamp("2019-05-31"), 200_000, 100_000, 1000)
    # gold held > 36 months, sold before July 2024: 20% with CII indexation
    H.record("G", "gold_etf", pd.Timestamp("2016-01-31"), pd.Timestamp("2019-05-31"), 150_000, 100_000, 1)
    expected = (150_000 - 100_000 * 289 / 254) * 0.20 * 1.04
    assert H.settle(2019) == pytest.approx(expected)


def test_debt_etf_after_april_2023_always_slab():
    T = E.TaxEngine(regime="historical", slab=0.20)
    T.record("D", "debt_etf", pd.Timestamp("2023-05-31"), pd.Timestamp("2026-05-29"), 130_000, 100_000, 1)
    assert T.settle(2026) == pytest.approx(30_000 * 0.20 * 1.04)


# ----------------------------------------------------------------------------- costs
def test_stt_stock_versus_equity_etf():
    C = E.CostModel(brokerage="zero")
    assert C.breakdown("buy", "stock", 10_000)["stt"] == pytest.approx(10.0)
    assert C.breakdown("buy", "eq_etf", 10_000)["stt"] == pytest.approx(0.0)
    assert C.breakdown("sell", "eq_etf", 10_000)["stt"] == pytest.approx(0.1)


def test_flat_fee_and_frictionless():
    flat = E.CostModel(brokerage="flat", flat_fee=20.0)
    assert flat.breakdown("buy", "gold_etf", 1_000)["brokerage"] == 20.0
    assert E.CostModel(frictionless=True).breakdown("sell", "stock", 1_000)["total"] == 0.0


# ----------------------------------------------------------------------------- simulator
@pytest.fixture
def prices():
    dates = pd.date_range("2018-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(1)
    px = 100 * np.exp(np.cumsum(rng.normal(0.008, 0.05, (60, 3)), axis=0))
    return pd.DataFrame(px, index=dates, columns=["E", "G", "D"])


def test_simulator_whole_units_and_accounting(prices):
    types = {"E": "eq_etf", "G": "gold_etf", "D": "debt_etf"}
    res = E.simulate(prices, types, lambda t: E.w_equal(prices.columns),
                     E.rebalance_schedule(prices.index, "A"), pd.Series(5000.0, index=prices.index),
                     E.CostModel(), E.TaxEngine(regime="current"), integer=True)
    assert res.total_contrib == pytest.approx(5000.0 * 60)
    assert res.total_cost > 0 and res.total_tax > 0
    assert res.terminal_post_tax < res.terminal_pre_tax
    assert (res.weights.sum(axis=1) <= 1.0 + 1e-9).all()        # no negative cash


def test_frictionless_matches_fractional_buy_and_hold(prices):
    """With no frictions and no rebalancing, a lump-sum single-asset portfolio tracks the price exactly."""
    one = prices[["E"]]
    res = E.simulate(one, {"E": "eq_etf"}, lambda t: pd.Series({"E": 1.0}), set(),
                     pd.Series([1000.0] + [0.0] * 59, index=one.index),
                     E.CostModel(frictionless=True), E.TaxEngine(regime="none"), integer=False)
    expected = (one["E"].pct_change().dropna())
    assert np.allclose(res.returns.values, expected.values)


# ----------------------------------------------------------------------------- weights and tests
def test_optimisers_respect_constraints():
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0.01, 0.05, (120, 5)), columns=list("abcde"))
    cov = E.cov_ledoit_wolf(R)
    for w in (E.w_minvar(cov, R.columns, 0.4), E.w_maxsharpe(R.mean().values, cov, 0.004, R.columns, 0.4),
              E.w_riskparity(cov, R.columns)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()
    assert E.w_minvar(cov, R.columns, 0.4).max() <= 0.4 + 1e-6


def test_ledoit_wolf_sharpe_test_same_population():
    """Two series from the same distribution should not differ significantly."""
    rng = np.random.default_rng(5)
    x = rng.normal(0.01, 0.04, 120)
    y = x + rng.normal(0, 0.01, 120)
    out = E.ledoit_wolf_sharpe_test(x, y, M=499)
    assert out["p_value"] > 0.05
    assert out["diff_annualised"] == pytest.approx(out["diff"] * np.sqrt(12))


def test_romano_wolf_adjusted_not_below_raw():
    rng = np.random.default_rng(7)
    b = rng.normal(0.01, 0.04, 120)
    others = {"x": b + rng.normal(0.003, 0.01, 120), "y": b + rng.normal(0, 0.01, 120)}
    rw = E.romano_wolf_sharpe(b, others, M=199).set_index("strategy")
    for k in others:
        raw = E.ledoit_wolf_sharpe_test(others[k], b, M=199)["p_value"]
        assert rw.loc[k, "rw_adj_p"] >= raw - 1e-9


# ----------------------------------------------------------------------------- data handling
def test_parse_both_bhavcopy_formats():
    def mkzip(csv):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("x.csv", csv)
        return b.getvalue()
    old = ("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,\n"
           "ABC,EQ,1,1,1,100,1,1,10,1000,30-JAN-2015,5,INE001,\n")
    new = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
           "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
           "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
           "2024-07-31,2024-07-31,CM,NSE,STK,1,INE001,ABC,EQ,,,,,ABC LTD,1,1,1,50,1,1,,50,,,10,500,3,F1,1,,,,,\n")
    a = D.parse_bhavcopy(mkzip(old), pd.Timestamp("2015-01-30"))
    b = D.parse_bhavcopy(mkzip(new), pd.Timestamp("2024-07-31"))
    assert list(a.columns) == list(b.columns)
    assert a["close"].iloc[0] == 100 and b["close"].iloc[0] == 50


def test_split_correction():
    c = pd.Series([100, 105, 52, 55, 11.2, 12], index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    r, n = D.clean_bhav_returns(c)
    assert n == 2
    assert r.abs().max() < 0.1


def test_nav_unit_normalisation_and_despike():
    d = pd.bdate_range("2019-11-01", periods=60)
    px = pd.Series(np.linspace(120, 130, 60), index=d)
    nav = px.copy()
    nav.iloc[:20] *= 10                       # NAV not adjusted for a 1:10 unit split
    out = D.normalize_nav_units(px, nav)
    assert np.allclose(out / px, 1.0)
    bad = px.copy()
    bad.iloc[[30, 31]] /= 10                  # two bad ticks
    clean, n = D.despike_daily(bad)
    assert n == 2 and (clean / px - 1).abs().max() < 0.02


def test_monthly_spike_reversal_cleaning():
    P = pd.DataFrame({"a": [100, 102, 300, 104, 106], "b": [50, 40, 30, 35, 36.0]})
    clean, n = D.clean_monthly_spikes(P)
    assert n == 1 and clean["a"].iloc[2] == pytest.approx(np.sqrt(102 * 104))
    assert clean["b"].equals(P["b"])
