# backend/app/utils.py
import time
import logging
from typing import Optional, List
import requests

# Optional fallback to yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger("fingrow.utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# FMP endpoints
FMP_QUOTE_URL = "https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={key}"
FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries={timeseries}&apikey={key}"
FMP_PROFILE = "https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={key}"

# common suffixes to try for Indian tickers
COMMON_SUFFIXES = ["", ".NS", ".BO"]

# Requests session with simple retry/backoff for resilience
def _requests_session_with_retries(total_retries=3, backoff_factor=0.3):
    session = requests.Session()
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    retries = Retry(total=total_retries,
                    backoff_factor=backoff_factor,
                    status_forcelist=(500, 502, 503, 504),
                    allowed_methods=frozenset(['GET', 'POST', 'OPTIONS']))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = _requests_session_with_retries()

def resolve_symbol(symbol: str, api_key: Optional[str]) -> str:
    """
    Try common suffixes and return the first symbol accepted by the FMP quote endpoint.
    If nothing validates, return the input uppercased.
    """
    if not symbol:
        raise ValueError("Empty symbol passed to resolve_symbol")
    s = symbol.strip().upper()
    if not api_key:
        return s
    for suf in COMMON_SUFFIXES:
        candidate = s if (suf == "" and "." in s) else s + suf
        try:
            url = FMP_QUOTE_URL.format(symbol=candidate, key=api_key)
            r = SESSION.get(url, timeout=6)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and len(j) and 'price' in j[0]:
                    logger.info("Resolved symbol %s -> %s via FMP", s, candidate)
                    return candidate
        except Exception:
            # ignore and try next candidate
            continue
    logger.info("Could not resolve symbol via FMP; using original symbol %s", s)
    return s

def fetch_current_price(symbol: str, api_key: Optional[str]) -> Optional[dict]:
    """
    Return a dict like {"price": <float>} or None.
    """
    if not api_key:
        return None
    try:
        sym = resolve_symbol(symbol, api_key)
        url = FMP_QUOTE_URL.format(symbol=sym, key=api_key)
        r = SESSION.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return {"price": data[0].get("price")}
    except Exception as e:
        logger.info("fetch_current_price failed for %s: %s", symbol, str(e))
        return None
    return None

def fetch_profile(symbol: str, api_key: Optional[str]) -> dict:
    """
    Fetch company profile from FMP; returns the profile dict or {} on failure.
    """
    if not api_key:
        return {}
    try:
        url = FMP_PROFILE.format(symbol=symbol, key=api_key)
        r = SESSION.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return j[0]
    except Exception as e:
        logger.info("fetch_profile failed for %s: %s", symbol, str(e))
    return {}

# ------------------------------
# Robust fetch_history function
# ------------------------------
def fetch_history(symbol: str, api_key: Optional[str], timeseries: int = 180) -> List[dict]:
    """
    Return a list of historical bars (dicts) newest-first (same shape as FMP).
    Strategy:
      1) Try FMP with descending timeseries sizes (timeseries, 365, 180, 90, 30).
      2) If FMP returns 403 or empty, try smaller windows.
      3) If FMP fails, fallback to yfinance (tries .NS/.BO/raw variants).
      4) If all fail, raise RuntimeError with a helpful message for the frontend.
    """
    tried = []
    # If we have an API key, try FMP first (descending windows)
    if api_key:
        sym = resolve_symbol(symbol, api_key)
        for ts in (timeseries, 365, 180, 90, 30):
            tried.append(ts)
            try:
                url = FMP_HIST_URL.format(symbol=sym, timeseries=ts, key=api_key)
                r = SESSION.get(url, timeout=18)
                if r.status_code == 200:
                    j = r.json()
                    hist = j.get("historical") or []
                    if hist:
                        logger.info("Fetched %d historical rows from FMP for %s (ts=%d)", len(hist), sym, ts)
                        # Ensure newest-first; FMP commonly returns newest-first already
                        return hist
                    # no rows — try smaller window
                    logger.info("FMP returned empty historical for %s (ts=%d)", sym, ts)
                    time.sleep(0.12)
                    continue
                elif r.status_code == 403:
                    logger.warning("FMP returned 403 for %s (ts=%d) — trying smaller window", sym, ts)
                    time.sleep(0.12)
                    continue
                else:
                    logger.warning("FMP returned status %s for %s (ts=%d); trying next", r.status_code, sym, ts)
                    time.sleep(0.12)
                    continue
            except requests.RequestException as rexc:
                logger.info("Network error fetching FMP for %s (ts=%d): %s", sym, ts, str(rexc))
                time.sleep(0.15)
                continue

    # FMP failed or no API key — try yfinance fallback if available
    if YFINANCE_AVAILABLE:
        try:
            yf_hist = _fetch_history_yfinance(symbol, period="2y", interval="1d")
            if yf_hist:
                logger.info("Using yfinance fallback for %s; rows=%d", symbol, len(yf_hist))
                return yf_hist
        except Exception as e:
            logger.info("yfinance fallback failed for %s: %s", symbol, str(e))

    # Nothing worked
    raise RuntimeError(
        f"Unable to fetch historical data for {symbol}. Tried FMP timeseries: {tried}. "
        "This usually means your FMP API key or plan doesn't allow historical endpoints or you hit rate limits. "
        "Try providing a different API key in the app or use a paid plan."
    )

def _fetch_history_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> List[dict]:
    """
    Robust yfinance fallback that tries multiple symbol variants.
    Returns list of dicts newest-first or [] if nothing found.
    """
    if not YFINANCE_AVAILABLE:
        return []

    s = symbol.strip().upper()
    candidates = []
    if "." in s:
        candidates = [s, s.replace(".NS", ""), s.replace(".BO", "")]
    else:
        candidates = [s + ".NS", s + ".BO", s, s + ".NSE"]
    candidates = list(dict.fromkeys(candidates))  # remove duplicates preserve order

    for cand in candidates:
        try:
            t = yf.Ticker(cand)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            if df is None or df.empty:
                logger.info("yfinance empty for %s", cand)
                continue
            df = df.reset_index()
            rows = []
            for _, r in df.iterrows():
                date_val = r.get('Date') or r.get('date') or None
                if hasattr(date_val, 'strftime'):
                    date = date_val.strftime("%Y-%m-%d")
                else:
                    date = str(date_val)
                rows.append({
                    "date": date,
                    "open": float(r.get('Open', 0) or 0),
                    "high": float(r.get('High', 0) or 0),
                    "low": float(r.get('Low', 0) or 0),
                    "close": float(r.get('Close', 0) or 0),
                    "volume": int(r.get('Volume', 0) or 0)
                })
            if rows:
                # yfinance produced oldest-first; convert to newest-first to match FMP usage
                return list(reversed(rows))
        except Exception as e:
            logger.info("yfinance candidate %s failed: %s", cand, str(e))
            continue

    return []
