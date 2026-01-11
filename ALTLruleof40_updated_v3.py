"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.
"""

from __future__ import annotations

# Rule of 40 Stock Scanner (Python)
#
# Scans Large and Mid cap companies in the S&P 500 or Nasdaq universe
#
# Rule of 40 score = Revenue Growth (YoY, TTM) + Free Cash Flow Margin (TTM)
#
# Where:
# - Revenue Growth (YoY, TTM) = (Revenue_TTM / Revenue_TTM_prev_year_TTM) - 1
# - FCF Margin (TTM)          = FCF_TTM / Revenue_TTM
# - FCF_TTM                   = Operating Cash Flow_TTM - |CapEx_TTM|
#
# Data source: Yahoo Finance via yfinance.
#
# Practical notes:
# - Rule of 40 is most meaningful for operating companies (e.g., Software / Tech),
#   and is often misleading for Financials, Utilities, REITs, and Energy (cashflow mechanics differ).
# - Yahoo data can be missing/inconsistent for some tickers; this script prioritizes "high confidence"
#   calculations using quarterly statements and skips lower-confidence rows by default.

import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: yfinance\n"
        "Install with:\n"
        "  pip install yfinance pandas\n"
    ) from e


# ----------------------------
# Configuration (edit these)
# ----------------------------

SLEEP_BETWEEN_TICKERS_SEC = 0.20  # be polite to Yahoo
RETRIES = 2
RETRY_SLEEP_SEC = 0.6

# Size filter (Mid + Large caps)
MIN_MARKET_CAP_USD = 2_000_000_000  # >= $2B

# Rule-of-40 filter
RULE_OF_40_THRESHOLD = 0.40  # >= 40%

# Confidence / sanity filters
# NOTE: Many tickers may be missing quarterly statements via Yahoo/yfinance at runtime.
# To avoid ending up with an empty result set, we *default* to allowing annual proxies,
# but we surface a `confidence` column (HIGH vs LOW) so you can decide what to trust.
REQUIRE_HIGH_CONFIDENCE = False        # if True, keeps only rows with quarterly revenue + quarterly FCF
LOW_CONFIDENCE_PENALTY = 0.10          # subtract 10 pts (0.10) from Rule40 when confidence is LOW
LOW_FCF_CONF_PENALTY = 0.05            # subtract 5 pts if FCF is annual proxy
MAX_ABS_FCF_MARGIN = 2.00              # skip if |FCF margin| > 200% (guards extreme anomalies)

# Sector filters (recommended ON; edit as desired)
EXCLUDED_SECTORS = {
    "Financial Services",
    "Utilities",
    "Energy",
    "Real Estate",
}
# If you prefer an allowlist, set INCLUDE_SECTORS to a set of sectors and leave EXCLUDED_SECTORS empty.
INCLUDE_SECTORS: Optional[set[str]] = None

# Optional strictness
REQUIRE_POSITIVE_REVENUE_GROWTH = False

# RSI (technical column)
RSI_PERIOD = 14
RSI_LOOKBACK = "6mo"  # amount of price history to compute RSI


# ----------------------------
# CSV / universe helpers
# ----------------------------

def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t.startswith("$"):
        t = t[1:]
    return t.replace(".", "-")


def ask_use_filters() -> bool:
    ans = input("Apply filters (CapBucket / Sector) from master CSV? (y/n): ").strip().lower()
    return ans in ("y", "yes")


def ask_cap_filter() -> Optional[set[str]]:
    print("\nMarket cap filter (CapBucket in master CSV):")
    print("  1) All")
    print("  2) Large")
    print("  3) Mid")
    print("  4) Small")
    print("  5) Micro")
    print("  6) Large+Mid")
    print("  7) Mid+Small")
    choice = input("Choose 1-7: ").strip()

    mapping = {
        "1": None,
        "2": {"Large"},
        "3": {"Mid"},
        "4": {"Small"},
        "5": {"Micro"},
        "6": {"Large", "Mid"},
        "7": {"Mid", "Small"},
    }
    if choice not in mapping:
        raise ValueError("Invalid cap filter choice.")
    return mapping[choice]


def ask_sector_filter(available_sectors: list[str]) -> Optional[set[str]]:
    print("\nSector filter (from master CSV):")
    print("  1) All")
    print("  2) Choose from list")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        return None
    if choice != "2":
        raise ValueError("Invalid sector filter choice.")

    if not available_sectors:
        print("No sector data available in this file. Sector filter skipped.")
        return None

    print("\nAvailable sectors:")
    for i, s in enumerate(available_sectors, 1):
        print(f"  {i}) {s}")

    raw = input("Enter sector numbers separated by commas (e.g. 1,3,7): ").strip()
    idx = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx.append(int(part))

    picked = {available_sectors[i - 1] for i in idx if 1 <= i <= len(available_sectors)}
    return picked if picked else None


def load_tickers_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "Ticker" in df.columns:
        series = df["Ticker"]
    elif "Symbol" in df.columns:
        series = df["Symbol"]
    else:
        series = df.iloc[:, 0]

    tickers = (
        series.astype(str)
        .map(normalize_ticker)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(set(tickers))


def load_master_with_filters(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        df["Ticker"] = df.iloc[:, 0]

    df["Ticker"] = df["Ticker"].astype(str).map(normalize_ticker)
    df = df.dropna(subset=["Ticker"])

    if not ask_use_filters():
        tickers = sorted(set(df["Ticker"].tolist()))
        print(f"\nLoaded {len(tickers)} tickers (no master filters)\n")
        return tickers

    cap_choice = None
    sector_choice = None

    if "CapBucket" in df.columns:
        df["CapBucket"] = df["CapBucket"].fillna("Unknown")
        cap_choice = ask_cap_filter()
    else:
        print("\nCapBucket column not found. Cap filter skipped.")

    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].fillna("Unknown")
        sectors = sorted([s for s in df["Sector"].unique().tolist() if s and s != "Unknown"])
        sector_choice = ask_sector_filter(sectors)
    else:
        print("\nSector column not found. Sector filter skipped.")

    if cap_choice is not None:
        df = df[df.get("CapBucket", "Unknown").isin(cap_choice)]
    if sector_choice is not None:
        df = df[df.get("Sector", "Unknown").isin(sector_choice)]

    tickers = sorted(set(df["Ticker"].tolist()))
    print(f"\nFiltered universe size (master filters): {len(tickers)}\n")
    return tickers


def ask_universe() -> tuple[str, list[str]]:
    print("\nSelect universe to scan:")
    print("  1) S&P 500")
    print("  2) Nasdaq")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        universe_name = "S&P 500"
        master = Path("sp500_master.csv")
        fallback = Path("sp500_companies.csv")
    elif choice == "2":
        universe_name = "Nasdaq"
        master = Path("nasdaq_master.csv")
        fallback = Path("nasdaq_tickers.csv")
    else:
        raise ValueError("Invalid choice. Please enter 1 or 2.")

    if master.exists():
        print(f"\nUsing master file: {master.name}")
        tickers = load_master_with_filters(master)
    else:
        if not fallback.exists():
            raise FileNotFoundError(
                f"Missing both {master.resolve()} and {fallback.resolve()}.\n"
                f"Create the master file (recommended) or place the fallback CSV in this folder."
            )
        print(f"\nMaster file not found. Using fallback list: {fallback.name} (no cap/sector master filters)")
        tickers = load_tickers_from_csv(fallback)
        print(f"Loaded {len(tickers)} tickers.\n")

    return universe_name, tickers


# ----------------------------
# Data helpers
# ----------------------------

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _retry(fn, *args, **kwargs):
    last = None
    for i in range(RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if i < RETRIES:
                time.sleep(RETRY_SLEEP_SEC)
    raise last  # type: ignore[misc]


def _sum_last_n(df: pd.DataFrame, row_name: str, n: int) -> Optional[float]:
    """
    Sum the last n columns for a given row from a yfinance statement.
    yfinance statements are typically indexed by line item with columns as dates.
    yfinance usually returns most-recent columns first.
    """
    if df is None or df.empty:
        return None
    if row_name not in df.index:
        return None
    series = df.loc[row_name]
    if series is None or len(series) < n:
        return None
    vals = pd.to_numeric(series.iloc[:n], errors="coerce")
    s = vals.sum()
    return _safe_float(s)


def _ttm_and_prev_year_ttm_revenue(t: yf.Ticker) -> Tuple[Optional[float], Optional[float], str]:
    """
    Returns (rev_ttm, rev_prev_year_ttm, method_used).

    Goal:
      - Use TRUE quarterly TTM revenue when available (sum of last 4 quarters)
      - Compute prior-year TTM revenue as the previous 4 quarters (sum of last 8 - sum of last 4)

    Yahoo line-item labels vary across tickers, so we search a small alias set and also
    fall back to a fuzzy match (case/space-insensitive).
    """
    REVENUE_ALIASES = [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
        "Net Sales",
        "Sales",
        "Total revenue",
        "Operating revenue",
    ]

    def _find_row(df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        # exact match first
        for r in REVENUE_ALIASES:
            if r in df.index:
                return r

        # case/space-insensitive map
        norm_map = {}
        for idx in df.index.astype(str):
            key = "".join(idx.lower().split())
            norm_map[key] = idx

        for r in REVENUE_ALIASES:
            key = "".join(r.lower().split())
            if key in norm_map:
                return norm_map[key]

        # fuzzy: contains 'revenue' and maybe 'total' preferred
        candidates = [str(i) for i in df.index if "revenue" in str(i).lower()]
        if not candidates:
            return None
        # prefer "total revenue" if present
        for c in candidates:
            if "total" in c.lower() and "revenue" in c.lower():
                return c
        return candidates[0]

    # Quarterly (preferred) — try both modern and legacy yfinance attributes
    q_inc = None
    try:
        q_inc = _retry(lambda: t.quarterly_income_stmt)
    except Exception:
        pass
    if (q_inc is None or getattr(q_inc, "empty", True)):
        try:
            q_inc = _retry(lambda: t.quarterly_financials)  # legacy
        except Exception:
            q_inc = None

    try:
        row = _find_row(q_inc) if q_inc is not None else None
        if row:
            rev_ttm = _sum_last_n(q_inc, row, 4)
            rev_8 = _sum_last_n(q_inc, row, 8)
            if rev_ttm is not None and rev_8 is not None:
                rev_prev_year = rev_8 - rev_ttm
                return rev_ttm, _safe_float(rev_prev_year), "quarterly_income_stmt"
    except Exception:
        pass

    # Annual fallback (proxy)
    a_inc = None
    try:
        a_inc = _retry(lambda: t.income_stmt)
    except Exception:
        pass
    if (a_inc is None or getattr(a_inc, "empty", True)):
        try:
            a_inc = _retry(lambda: t.financials)  # legacy
        except Exception:
            a_inc = None

    try:
        row = _find_row(a_inc) if a_inc is not None else None
        if row:
            rev_year0 = _sum_last_n(a_inc, row, 1)
            rev_year1 = _sum_last_n(a_inc, row, 2)
            if rev_year0 is not None and rev_year1 is not None:
                rev_prev_year = rev_year1 - rev_year0
                return rev_year0, _safe_float(rev_prev_year), "annual_income_stmt_proxy"
    except Exception:
        pass

    return None, None, "missing_revenue"


def _ttm_fcf(t: yf.Ticker) -> Tuple[Optional[float], str]:
    """
    Returns (fcf_ttm, method_used).
    FCF_TTM = OCF_TTM - |CapEx_TTM|
    """
    # Quarterly (preferred)
    try:
        q_cf = _retry(lambda: t.quarterly_cashflow)
        ocf_ttm = _sum_last_n(q_cf, "Operating Cash Flow", 4)
        capex_ttm = _sum_last_n(q_cf, "Capital Expenditure", 4)
        if ocf_ttm is not None and capex_ttm is not None:
            capex_ttm = abs(capex_ttm)  # IMPORTANT: capex often appears as negative
            return _safe_float(ocf_ttm - capex_ttm), "quarterly_cashflow"
    except Exception:
        pass

    # Annual fallback (proxy)
    try:
        a_cf = _retry(lambda: t.cashflow)
        ocf = _sum_last_n(a_cf, "Operating Cash Flow", 1)
        capex = _sum_last_n(a_cf, "Capital Expenditure", 1)
        if ocf is not None and capex is not None:
            capex = abs(capex)
            return _safe_float(ocf - capex), "annual_cashflow_proxy"
    except Exception:
        pass

    return None, "missing_fcf"


def _get_market_cap(t: yf.Ticker, info: Dict[str, Any]) -> Optional[float]:
    mc = None
    try:
        fast_info = getattr(t, "fast_info", None)
        if fast_info:
            mc = getattr(fast_info, "market_cap", None)
            if mc is None and isinstance(fast_info, dict):
                mc = fast_info.get("market_cap")
    except Exception:
        mc = None
    if mc is None:
        mc = info.get("marketCap")
    return _safe_float(mc)


def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> Optional[float]:
    if close is None or close.empty or len(close) <= period:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return _safe_float(rsi.iloc[-1])


def fetch_rsi_from_ticker(t: yf.Ticker, period: int = RSI_PERIOD) -> Optional[float]:
    try:
        hist = _retry(lambda: t.history(period=RSI_LOOKBACK, interval="1d"))
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return _compute_rsi(hist["Close"], period=period)
    except Exception:
        return None


# ----------------------------
# Row schema
# ----------------------------

@dataclass
class Rule40Row:
    ticker: str
    name: Optional[str]
    sector: Optional[str]
    market_cap: Optional[float]

    revenue_ttm: Optional[float]
    revenue_prev_year_ttm: Optional[float]
    revenue_growth_yoy: Optional[float]

    fcf_ttm: Optional[float]
    fcf_margin: Optional[float]
    rule_of_40: Optional[float]

    rsi: Optional[float]

    revenue_method: str
    fcf_method: str
    confidence: str

    error: Optional[str] = None


def _sector_allowed(sector: Optional[str]) -> bool:
    if sector is None:
        return True  # keep unknowns unless you want to drop them
    if INCLUDE_SECTORS is not None:
        return sector in INCLUDE_SECTORS
    if EXCLUDED_SECTORS:
        return sector not in EXCLUDED_SECTORS
    return True


# ----------------------------
# Core compute
# ----------------------------

def compute_rule_of_40(ticker: str) -> Rule40Row:
    try:
        t = yf.Ticker(ticker)

        # Metadata
        try:
            info = _retry(lambda: t.get_info() or {})
        except Exception:
            info = {}

        name = info.get("shortName") or info.get("longName")
        sector = info.get("sector")
        mcap = _get_market_cap(t, info)

        # Financials
        rev_ttm, rev_prev, rev_method = _ttm_and_prev_year_ttm_revenue(t)
        fcf_ttm, fcf_method = _ttm_fcf(t)

        # Technical
        rsi = fetch_rsi_from_ticker(t)

        # Derived metrics
        growth = None
        if rev_ttm is not None and rev_prev is not None and rev_prev != 0:
            growth = (rev_ttm / rev_prev) - 1.0

        fcf_margin = None
        if fcf_ttm is not None and rev_ttm is not None and rev_ttm != 0:
            fcf_margin = fcf_ttm / rev_ttm

        rule40 = None
        if growth is not None and fcf_margin is not None:
            rule40 = growth + fcf_margin

        confidence = "HIGH" if (rev_method == "quarterly_income_stmt" and fcf_method == "quarterly_cashflow") else "LOW"

        return Rule40Row(
            ticker=ticker.upper(),
            name=name,
            sector=sector,
            market_cap=mcap,
            revenue_ttm=rev_ttm,
            revenue_prev_year_ttm=rev_prev,
            revenue_growth_yoy=growth,
            fcf_ttm=fcf_ttm,
            fcf_margin=fcf_margin,
            rule_of_40=rule40,
            rsi=rsi,
            revenue_method=rev_method,
            fcf_method=fcf_method,
            confidence=confidence,
            error=None,
        )

    except Exception as e:
        return Rule40Row(
            ticker=ticker.upper(),
            name=None,
            sector=None,
            market_cap=None,
            revenue_ttm=None,
            revenue_prev_year_ttm=None,
            revenue_growth_yoy=None,
            fcf_ttm=None,
            fcf_margin=None,
            rule_of_40=None,
            rsi=None,
            revenue_method="error",
            fcf_method="error",
            confidence="LOW",
            error=str(e),
        )


def scan(tickers: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for tk in tickers:
        tk = tk.strip().upper()
        if not tk:
            continue

        row = compute_rule_of_40(tk)

        # Filters
        if row.error:
            # Skip hard errors; you can keep them if you want diagnostics
            continue

        if row.market_cap is not None and row.market_cap < MIN_MARKET_CAP_USD:
            continue

        if not _sector_allowed(row.sector):
            continue

        # Confidence gating (optional)
        if REQUIRE_HIGH_CONFIDENCE and row.confidence != "HIGH":
            continue

        if row.fcf_margin is None:
            continue
        if abs(row.fcf_margin) > MAX_ABS_FCF_MARGIN:
            continue

        if REQUIRE_POSITIVE_REVENUE_GROWTH and (row.revenue_growth_yoy is None or row.revenue_growth_yoy <= 0):
            continue

        if row.rule_of_40 is None or row.rule_of_40 < RULE_OF_40_THRESHOLD:
            continue

        rows.append(asdict(row))
        time.sleep(SLEEP_BETWEEN_TICKERS_SEC)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Numeric conversions + friendly % columns
    for col in ["market_cap", "revenue_growth_yoy", "fcf_margin", "rule_of_40", "rsi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["revenue_growth_yoy_pct"] = (df["revenue_growth_yoy"] * 100).round(2)
    df["fcf_margin_pct"] = (df["fcf_margin"] * 100).round(2)
    df["rule_of_40_pct"] = (df["rule_of_40"] * 100).round(2)
    df["rsi"] = df["rsi"].round(2)

    # Confidence-weighted score (keeps list populated but nudges LOW-confidence names down)
    df["rule_of_40_adj"] = df["rule_of_40"]
    df.loc[df["confidence"] != "HIGH", "rule_of_40_adj"] = df.loc[df["confidence"] != "HIGH", "rule_of_40_adj"] - LOW_CONFIDENCE_PENALTY
    df.loc[df["fcf_method"] == "annual_cashflow_proxy", "rule_of_40_adj"] = df.loc[df["fcf_method"] == "annual_cashflow_proxy", "rule_of_40_adj"] - LOW_FCF_CONF_PENALTY
    df["rule_of_40_adj_pct"] = (df["rule_of_40_adj"] * 100).round(2)

    # Sort by adjusted score (then raw Rule-of-40)
    df = df.sort_values(["rule_of_40_adj", "rule_of_40"], ascending=[False, False])

    return df


if __name__ == "__main__":
    try:
        universe_name, tickers = ask_universe()
    except Exception as e:
        print(f"\nError selecting universe: {e}")
        sys.exit(1)

    if not tickers:
        raise SystemExit("No tickers loaded; aborting scan.")

    print(
        f"\nScanning {len(tickers)} tickers from {universe_name} "
        f"(>=${MIN_MARKET_CAP_USD/1e9:.0f}B, Rule40>={RULE_OF_40_THRESHOLD*100:.0f}%)..."
    )
    if EXCLUDED_SECTORS:
        print(f"Excluded sectors: {sorted(EXCLUDED_SECTORS)}")
    if INCLUDE_SECTORS is not None:
        print(f"Included sectors (allowlist): {sorted(INCLUDE_SECTORS)}")

    df = scan(tickers)
    if df.empty:
        print("\nNo tickers met the filters. Tips:")
        print("- If you set REQUIRE_HIGH_CONFIDENCE=True, try switching it to False to allow annual proxy statements.")
        print("- Relax MAX_ABS_FCF_MARGIN (e.g., 3.0) if too strict, or disable sector exclusions.")
        print("- Check your internet / Yahoo rate limits; increase SLEEP_BETWEEN_TICKERS_SEC if many missing rows.")
        sys.exit(0)

    cols = [
        "ticker", "name", "sector", "market_cap",
        "revenue_growth_yoy_pct", "fcf_margin_pct", "rule_of_40_pct", "rule_of_40_adj_pct",
        "rsi", "confidence", "revenue_method", "fcf_method",
    ]
    cols = [c for c in cols if c in df.columns]

    print("\nResults:")
    print(df[cols].to_string(index=False))

    out_csv = "rule_of_40_scan.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
