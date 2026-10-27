"""
data_io.py
==========
Free-data acquisition for the ETF small-investor study, with on-disk caching and a
download manifest (URL, file, timestamp, SHA-256, row count) for reproducibility.

Free sources used
-----------------
* Yahoo Finance via `yfinance`  : ETF and stock daily prices (Close, Adj Close, Volume)
* NSE archives (bhavcopy)       : month-end snapshots of all listed equities, including
                                  stocks later delisted -> survivorship-bias-free universe
* mfapi.in (mirror of AMFI NAVs): daily NAV of each ETF scheme and of a liquid fund
* FRED (St. Louis Fed)          : India 3-month Treasury bill rate (INTGSTINM193N)
* niftyindices.com              : Total Return Index (TRI) of NIFTY indices

Every function caches its raw output under data/raw/... and never re-downloads a file
that already exists unless `force=True`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import time
import zipfile
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Append-only JSON-lines log of every file downloaded or derived."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, source: str, url: str, file: str, rows: Optional[int] = None, note: str = ""):
        rec = {"utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "source": source,
               "url": url, "file": file, "rows": rows, "note": note,
               "sha256": sha256(file) if file and os.path.exists(file) else None}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def table(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        return pd.read_json(self.path, lines=True)


def http_get(url: str, session=None, retries: int = 3, timeout: int = 40, pause: float = 1.0,
             method: str = "GET", **kw) -> Optional[bytes]:
    """GET/POST with retries. Returns None on 404 (e.g. a market holiday)."""
    s = session or requests.Session()
    for k in range(retries):
        try:
            r = s.request(method, url, headers={**UA, **kw.pop("headers", {})}, timeout=timeout, **kw)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
        time.sleep(pause * (k + 1))
    return None


# -----------------------------------------------------------------------------
# Yahoo Finance
# -----------------------------------------------------------------------------
def yf_history(tickers: List[str], start: str, end: str, batch: int = 50) -> pd.DataFrame:
    """Daily OHLCV with both Close and Adj Close, long format."""
    import yfinance as yf
    out = []
    for i in range(0, len(tickers), batch):
        tk = tickers[i:i + batch]
        df = yf.download(tk, start=start, end=end, auto_adjust=False, actions=False,
                         group_by="ticker", progress=False, threads=True)
        if df is None or df.empty:
            continue
        for t in tk:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            sub = sub.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
            sub = sub.reset_index().rename(columns={"Date": "date", "date": "date"})
            sub["ticker"] = t
            out.append(sub[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]])
        time.sleep(1.0)
    if not out:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    res = pd.concat(out, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"]).dt.tz_localize(None)
    return res


# -----------------------------------------------------------------------------
# mfapi.in (AMFI NAV history)
# -----------------------------------------------------------------------------
def mfapi_search(query: str) -> List[Dict]:
    raw = http_get("https://api.mfapi.in/mf/search", params={"q": query})
    return json.loads(raw) if raw else []


def mfapi_nav(code: int) -> pd.DataFrame:
    raw = http_get(f"https://api.mfapi.in/mf/{code}")
    if not raw:
        return pd.DataFrame(columns=["date", "nav"])
    js = json.loads(raw)
    df = pd.DataFrame(js.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["date", "nav"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["scheme_name"] = js.get("meta", {}).get("scheme_name", "")
    return df[["date", "nav"]]


def pick_scheme(query: str, must: Iterable[str] = (), avoid: Iterable[str] = ()) -> Optional[Dict]:
    """Pick the first mfapi scheme whose name contains all `must` words and none of `avoid`."""
    for item in mfapi_search(query):
        name = item.get("schemeName", "").lower()
        if all(m.lower() in name for m in must) and not any(a.lower() in name for a in avoid):
            return item
    return None


# -----------------------------------------------------------------------------
# FRED
# -----------------------------------------------------------------------------
def fred_series(series_id: str) -> pd.DataFrame:
    raw = http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    if not raw:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


# -----------------------------------------------------------------------------
# niftyindices.com Total Return Index
# -----------------------------------------------------------------------------
def nifty_tri(index_name: str, start: str, end: str) -> pd.DataFrame:
    """Daily TRI from niftyindices.com, fetched one calendar year at a time."""
    s = requests.Session()
    s.headers.update({**UA, "Content-Type": "application/json; charset=UTF-8",
                      "Origin": "https://www.niftyindices.com",
                      "Referer": "https://www.niftyindices.com/reports/historical-data"})
    try:
        s.get("https://www.niftyindices.com/reports/historical-data", timeout=30)
    except Exception:
        pass
    url = "https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
    rows = []
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    for y in range(y0, y1 + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).strftime("%d-%b-%Y")
        b = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31")).strftime("%d-%b-%Y")
        cinfo = "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (index_name, a, b, index_name)
        try:
            r = s.post(url, data=json.dumps({"cinfo": cinfo}), timeout=40)
            js = json.loads(json.loads(r.text)["d"])
            for rec in js:
                rows.append({"date": pd.to_datetime(rec["Date"]),
                             "tri": float(str(rec["TotalReturnsIndex"]).replace(",", ""))})
        except Exception:
            pass
        time.sleep(1.0)
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["date", "tri"])


# -----------------------------------------------------------------------------
# NSE bhavcopy (month-end snapshots)
# -----------------------------------------------------------------------------
NSE_HOSTS = ["https://nsearchives.nseindia.com", "https://archives.nseindia.com"]
UDIFF_START = pd.Timestamp("2024-07-08")


def nse_session():
    s = requests.Session()
    s.headers.update({**UA, "Referer": "https://www.nseindia.com/"})
    try:
        s.get("https://www.nseindia.com", timeout=20)
    except Exception:
        pass
    return s


def _bhav_urls(d: pd.Timestamp) -> List[str]:
    mon = d.strftime("%b").upper()
    old = f"/content/historical/EQUITIES/{d.year}/{mon}/cm{d.strftime('%d')}{mon}{d.year}bhav.csv.zip"
    new = f"/content/cm/BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
    paths = [new, old] if d >= UDIFF_START else [old, new]
    return [h + p for p in paths for h in NSE_HOSTS]


def parse_bhavcopy(content: bytes, d: pd.Timestamp) -> pd.DataFrame:
    """Standardise old (cmDDMONYYYYbhav) and new (UDiFF) formats."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name))
    df.columns = [c.strip() for c in df.columns]
    if "SYMBOL" in df.columns:
        out = pd.DataFrame({"symbol": df["SYMBOL"], "series": df["SERIES"].astype(str).str.strip(),
                            "isin": df.get("ISIN"), "close": df["CLOSE"],
                            "volume": df["TOTTRDQTY"], "traded_value": df["TOTTRDVAL"]})
    else:
        out = pd.DataFrame({"symbol": df["TckrSymb"], "series": df["SctySrs"].astype(str).str.strip(),
                            "isin": df.get("ISIN"), "close": df["ClsPric"],
                            "volume": df["TtlTradgVol"], "traded_value": df["TtlTrfVal"]})
    out["date"] = d
    return out


def fetch_bhavcopy(d: pd.Timestamp, session) -> Optional[pd.DataFrame]:
    for url in _bhav_urls(d):
        raw = http_get(url, session=session, retries=2, pause=0.8)
        if raw and raw[:2] == b"PK":
            try:
                df = parse_bhavcopy(raw, d)
                df.attrs["url"] = url
                return df
            except Exception:
                continue
    return None


def month_end_bhavcopies(start: str, end: str, out_dir: str, manifest: Manifest,
                         force: bool = False, max_back: int = 10) -> pd.DataFrame:
    """Download the last trading day's bhavcopy of every month in [start, end]."""
    os.makedirs(out_dir, exist_ok=True)
    session = nse_session()
    frames = []
    months = pd.date_range(start, end, freq="ME")
    for m in months:
        fpath = os.path.join(out_dir, f"bhav_{m.strftime('%Y%m')}.csv.gz")
        if os.path.exists(fpath) and not force:
            frames.append(pd.read_csv(fpath, parse_dates=["date"]))
            continue
        got = None
        for back in range(max_back):
            d = m - pd.Timedelta(days=back)
            if d.weekday() >= 5:
                continue
            got = fetch_bhavcopy(d, session)
            if got is not None:
                break
            time.sleep(0.5)
        if got is None:
            print(f"  ! no bhavcopy found for {m:%Y-%m}")
            continue
        url = got.attrs.get("url", "")
        got.to_csv(fpath, index=False, compression="gzip")
        manifest.log("NSE bhavcopy", url, fpath, len(got))
        frames.append(got)
        print(f"  bhavcopy {got['date'].iloc[0]:%Y-%m-%d}: {len(got)} rows")
        time.sleep(1.0)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nifty500_current_list() -> pd.DataFrame:
    """Fallback universe (survivorship-biased) if NSE archives are unreachable."""
    raw = http_get("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv")
    return pd.read_csv(io.BytesIO(raw)) if raw else pd.DataFrame()


# -----------------------------------------------------------------------------
# Stock panel construction
# -----------------------------------------------------------------------------
SPLIT_FACTORS = np.array([1.5, 2, 2.5, 3, 4, 5, 6, 10, 11, 20])


def clean_bhav_returns(close: pd.Series, tol: float = 0.06):
    """Monthly returns from raw closes, correcting splits/bonuses detected as
    price ratios near 1/k for common k. Returns (returns, n_corrections)."""
    r = close / close.shift(1)
    out = r.copy()
    n = 0
    for i, x in r.items():
        if not np.isfinite(x) or x > 0.6:
            continue
        k = 1 / x
        j = np.argmin(np.abs(SPLIT_FACTORS - k))
        if abs(SPLIT_FACTORS[j] - k) / SPLIT_FACTORS[j] < tol:
            out[i] = x * SPLIT_FACTORS[j]
            n += 1
    return out - 1, n


def build_stock_panel(bhav: pd.DataFrame, yf_monthly_adj: pd.DataFrame, top_n: int = 500,
                      min_price: float = 10.0, lookback: int = 3, min_yf_coverage: float = 0.9):
    """Return (prices_wide, eligible_wide, meta).

    bhav           : long month-end bhavcopy rows (date, symbol, series, isin, close, traded_value)
    yf_monthly_adj : wide month-end Adj Close, columns = Yahoo tickers (SYMBOL.NS)
    """
    b = bhav[bhav["series"].isin(["EQ", "BE", "BZ"])].copy()
    b["date"] = pd.to_datetime(b["date"]).dt.to_period("M").dt.to_timestamp("M")
    b["id"] = b["isin"].fillna(b["symbol"])
    b = b.sort_values("date").drop_duplicates(["date", "id"], keep="last")
    latest_sym = b.groupby("id")["symbol"].last()
    tv = b.pivot(index="date", columns="id", values="traded_value")
    close = b.pivot(index="date", columns="id", values="close")
    eq_only = b[b["series"] == "EQ"].pivot(index="date", columns="id", values="close").notna()
    tv_avg = tv.rolling(lookback, min_periods=1).mean()
    rank = tv_avg.where(eq_only & (close >= min_price)).rank(axis=1, ascending=False)
    eligible = (rank <= top_n)

    ids = eligible.columns[eligible.any()]
    prices = {}
    meta = []
    for i in ids:
        sym = latest_sym[i]
        tk = f"{sym}.NS"
        bh = close[i].dropna()
        src, ncorr = "bhav", 0
        if tk in yf_monthly_adj.columns:
            y = yf_monthly_adj[tk].reindex(bh.index)
            cov = y.notna().mean() if len(bh) else 0
            if cov >= min_yf_coverage:
                prices[i] = yf_monthly_adj[tk].reindex(close.index).where(close[i].notna())
                src = "yahoo_adj"
        if src == "bhav":
            r, ncorr = clean_bhav_returns(bh)
            lvl = (1 + r.fillna(0)).cumprod() * bh.iloc[0]
            prices[i] = lvl.reindex(close.index)
        meta.append({"id": i, "symbol": sym, "price_source": src, "split_corrections": ncorr,
                     "first": bh.index.min(), "last": bh.index.max(),
                     "months_eligible": int(eligible[i].sum())})
    P = pd.DataFrame(prices).reindex(columns=ids)
    meta = pd.DataFrame(meta)
    global LAST_RANK
    LAST_RANK = rank[ids]
    return P, eligible[ids], meta


# -----------------------------------------------------------------------------
# Synthetic data (smoke test only; never used for results)
# -----------------------------------------------------------------------------
def synthetic_dataset(etf_keys: List[str], start="2015-01-01", end="2026-09-30", n_stocks=700, seed=7):
    """Generate data with the same structure as the live pipeline's processed outputs."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T = len(days)
    spec = {"eq_large": (0.12, 0.17), "eq_next50": (0.13, 0.21), "gold": (0.10, 0.14),
            "debt": (0.065, 0.04), "intl": (0.15, 0.22), "eq_mid": (0.15, 0.22)}
    k = len(etf_keys)
    C = np.full((k, k), 0.2)
    for a in range(k):
        for b_ in range(k):
            ka, kb = etf_keys[a], etf_keys[b_]
            if ka.startswith("eq") and kb.startswith("eq"):
                C[a, b_] = 0.85
            if "gold" in (ka, kb):
                C[a, b_] = -0.05
            if "debt" in (ka, kb):
                C[a, b_] = 0.05
        C[a, a] = 1
    L = np.linalg.cholesky(C + 1e-6 * np.eye(k))
    mu = np.array([spec.get(x, (0.1, 0.15))[0] for x in etf_keys]) / 252
    sd = np.array([spec.get(x, (0.1, 0.15))[1] for x in etf_keys]) / np.sqrt(252)
    Z = rng.standard_normal((T, k)) @ L.T
    R = mu + sd * Z
    tri = pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=days, columns=etf_keys)
    nav = tri * np.exp(-np.arange(T)[:, None] * 0.0005 / 252)       # expense drag
    nav = nav * (1 + rng.normal(0, 0.0003, nav.shape))                 # tracking noise
    price = nav * (1 + rng.normal(0, 0.002, nav.shape))                # premium/discount
    vol = pd.DataFrame(rng.lognormal(12, 1, (T, k)), index=days, columns=etf_keys)
    etf_daily = []
    for key in etf_keys:
        df = pd.DataFrame({"date": days, "key": key, "close": price[key].values,
                           "adj_close": price[key].values, "volume": vol[key].values.round()})
        etf_daily.append(df)
    etf_daily = pd.concat(etf_daily, ignore_index=True)
    nav_long = nav.stack().rename("nav").reset_index().rename(columns={"level_0": "date", "level_1": "key"})
    tri_long = tri.stack().rename("tri").reset_index().rename(columns={"level_0": "date", "level_1": "key"})
    months = pd.date_range(start, end, freq="ME")
    rf = pd.DataFrame({"date": months, "rf": 0.065 / 12 + rng.normal(0, 0.0003, len(months))})
    # stocks: one-factor model, random listing / delisting
    m = len(months)
    mkt = rng.normal(0.010, 0.05, m)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    idio = rng.normal(0, 1, (m, n_stocks)) * rng.uniform(0.06, 0.14, n_stocks)
    ret = 0.001 + beta * mkt[:, None] + idio
    lvl = rng.uniform(50, 3000, n_stocks) * np.cumprod(1 + ret, axis=0)
    P = pd.DataFrame(lvl, index=months, columns=[f"SYN{i:04d}" for i in range(n_stocks)])
    first = rng.integers(0, m // 2, n_stocks) * (rng.random(n_stocks) < 0.25)
    last = np.where(rng.random(n_stocks) < 0.12, rng.integers(m // 3, m, n_stocks), m)
    for j, c in enumerate(P.columns):
        P.iloc[:first[j], j] = np.nan
        P.iloc[last[j]:, j] = np.nan
    tv = P.notna() * rng.lognormal(18, 1.5, n_stocks)
    rank = tv.where(P.notna()).rank(axis=1, ascending=False)
    elig = rank <= 500
    meta = pd.DataFrame({"id": P.columns, "symbol": P.columns, "price_source": "synthetic",
                         "split_corrections": 0})
    return {"etf_daily": etf_daily, "etf_nav": nav_long, "etf_tri": tri_long, "rf": rf,
            "stock_prices": P, "stock_eligible": elig, "stock_rank": rank, "stock_meta": meta}


TRI_DIAG: Dict[str, dict] = {}
LAST_RANK = None


def read_tri_files(raw_dir: str, index_name: str, verbose: bool = True, keep_file: bool = False) -> pd.DataFrame:
    """Read TRI CSVs saved manually from niftyindices.com (Historical Data -> Total Returns Index).
    Accepts one or many files named like NIFTY_50*.csv; any column containing 'date' and any
    column containing 'total' (or named 'tri') are used."""
    import glob
    stem = index_name.replace(" ", "_")
    frames, diag = [], []
    files = sorted(set(glob.glob(os.path.join(raw_dir, f"{stem}_*.csv")) + glob.glob(os.path.join(raw_dir, f"{stem}.csv"))))
    for f in files:   # "NIFTY_50_*.csv" deliberately does not match NIFTY_500 files
        if index_name == "NIFTY 50" and os.path.basename(f).upper().startswith("NIFTY_50_EQUAL"):
            continue
        df = pd.read_csv(f)
        cols = {c.lower().strip(): c for c in df.columns}
        dc = next((c for k, c in cols.items() if "date" in k), None)
        vc = next((c for k, c in cols.items() if "total" in k or k == "tri"), None)
        if dc is None or vc is None:
            print(f"  skipped {os.path.basename(f)}: needs a date column and a Total Returns column")
            continue
        one = pd.DataFrame({
            "date": pd.to_datetime(df[dc], dayfirst=True, errors="coerce"),
            "tri": pd.to_numeric(df[vc].astype(str).str.replace(",", ""), errors="coerce")}).dropna()
        one["file"] = os.path.basename(f)
        name_col = next((c for k, c in cols.items() if "index" in k and "name" in k), None)
        diag.append({"file": os.path.basename(f), "rows": len(one),
                     "first_date": one["date"].min(), "last_date": one["date"].max(),
                     "first_value": one.sort_values("date")["tri"].iloc[0] if len(one) else np.nan,
                     "last_value": one.sort_values("date")["tri"].iloc[-1] if len(one) else np.nan,
                     "index_name_in_file": str(df[name_col].iloc[0]) if name_col and len(df) else "",
                     "value_column": vc})
        frames.append(one)
    if not frames:
        return pd.DataFrame(columns=["date", "tri"])
    out = pd.concat(frames).sort_values("date")
    dups = out[out.duplicated("date", keep=False)]
    conflict = dups.groupby("date")["tri"].nunique()
    out = out.drop_duplicates("date").reset_index(drop=True)
    r = out["tri"].pct_change()
    jumps = out.loc[r.abs() > 0.15, ["date", "tri", "file"]].assign(daily_return=r[r.abs() > 0.15])
    if verbose:
        print(f"  {index_name}: files read")
        print(pd.DataFrame(diag).to_string(index=False))
        if (conflict > 1).any():
            print(f"  ! {int((conflict > 1).sum())} dates appear in two files with DIFFERENT values (overlapping downloads).")
        if len(jumps):
            print(f"  ! {len(jumps)} day(s) with a TRI move above 15% - almost always a wrong file "
                  f"(price index instead of TRI, another index, or a mis-read date):")
            print(jumps.to_string(index=False))
        else:
            print("  OK: no daily TRI move above 15% (largest genuine move: about -13% on 23-Mar-2020).")
    TRI_DIAG[index_name] = {"files": pd.DataFrame(diag), "jumps": jumps}   # not in .attrs: pandas concat breaks on DataFrame attrs
    return out[["date", "tri"]].assign(file=out["file"]) if keep_file else out[["date", "tri"]]


def normalize_nav_units(price: pd.Series, nav: pd.Series, tol: float = 0.20) -> pd.Series:
    """Put NAV on the same per-unit basis as the (split-adjusted) market price.

    Yahoo prices are back-adjusted for unit splits; AMFI NAVs are not. On each day the
    NAV/price ratio is snapped to the nearest split factor (1, 2, 5, 10, ... 1000); if it is
    within `tol` of a factor other than 1, NAV is divided by that factor."""
    df = pd.concat([price.rename("p"), nav.rename("n")], axis=1).dropna()
    if df.empty:
        return nav
    facs = np.array(sorted({a * 10 ** b for a in (1, 2, 5) for b in range(-3, 4)}))
    # 21-day centred median of the NAV/price ratio: robust to genuine premiums (e.g. Nasdaq ETF
    # trading 5-25% above NAV) and to isolated bad ticks; snapped to the nearest split factor.
    lk = np.log(df["n"] / df["p"]).rolling(21, center=True, min_periods=3).median().values
    idx = np.abs(lk[:, None] - np.log(facs)[None, :]).argmin(axis=1)
    c = facs[idx]
    ok = (np.abs(lk - np.log(c)) < np.log(1.4)) & (c != 1)
    adj = pd.Series(np.where(ok, c, 1.0), index=df.index)
    out = nav.copy()
    common = out.index.intersection(adj.index)
    out.loc[common] = out.loc[common] / adj.loc[common]
    # days with NAV but no price: carry the nearest factor
    rest = out.index.difference(adj.index)
    if len(rest):
        f = adj.reindex(out.index).ffill().bfill()
        out.loc[rest] = nav.loc[rest] / f.loc[rest]
    return out


def despike_daily(s: pd.Series, window: int = 11, thresh: float = 0.40):
    """Remove bad ticks from a daily price series: points more than `thresh` (40%) away from the
    centred rolling median of `window` days are set missing and interpolated. A diversified ETF
    cannot move 40% against its own two-week median and back. Returns (clean series, n days fixed)."""
    x = s.dropna()
    if len(x) < window:
        return s, 0
    m = x.rolling(window, center=True, min_periods=3).median()
    bad = (x / m - 1).abs() > thresh
    y = x.mask(bad).interpolate(method="time").ffill().bfill()
    out = s.copy()
    out.loc[y.index] = y
    return out, int(bad.sum())


def clean_monthly_spikes(P: pd.DataFrame, jump: float = 0.5, back: float = 0.2):
    """Fix one-month price spikes that reverse the next month (classic data error): if
    |r_t| > `jump` and the two-month return (1+r_t)(1+r_t+1)-1 is within +-`back`, the month-t
    price is replaced by the geometric mean of its neighbours. Returns (clean panel, count)."""
    r = P.pct_change(fill_method=None)
    two = (1 + r) * (1 + r.shift(-1)) - 1
    flag = (r.abs() > jump) & (two.abs() < back)
    fix = np.sqrt(P.shift(1) * P.shift(-1))
    return P.mask(flag, fix), int(flag.sum().sum())
