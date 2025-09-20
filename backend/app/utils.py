# backend/app/utils.py
import time
import logging
from typing import Optional, List, Tuple
from datetime import date, timedelta

import requests

# Optional fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

try:
    import nsepy  # for availability check
    NSEPY_AVAILABLE = True
except Exception:
    NSEPY_AVAILABLE = False

# Setup logger
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

COMMON_SUFFIXES = ["", ".NS", ".BO"]

# Requests session with retries
def _requests_session_with_retries(total_retries=3, backoff_factor=0.3):
    session = requests.Session()
    try:
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retries = Retry(
            total=total_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(['GET', 'POST', 'OPTIONS'])
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception:
        pass
    return session

SESSION = _requests_session_with_retries()


def resolve_symbol(symbol: str, api_key: Optional[str]) -> str:
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
                    logger.info("Resolved %s -> %s via FMP", s, candidate)
                    return candidate
        except Exception:
            continue
    logger.info("Could not resolve %s via FMP; using original", s)
    return s


def fetch_current_price(symbol: str, api_key: Optional[str]) -> Optional[dict]:
    """
    Try to get current price with fallbacks.
    Order: FMP -> yfinance -> NSEpy.
    Returns {"price": float, "source": "<provider>"} or None.
    """
    s_raw = (symbol or "").strip()
    if not s_raw:
        logger.info("fetch_current_price: empty symbol")
        return None
    s = s_raw.upper()
    logger.info("fetch_current_price: start for %s", s)

    # 1) FMP
    if api_key:
        try:
            sym = resolve_symbol(s, api_key)
            url = FMP_QUOTE_URL.format(symbol=sym, key=api_key)
            r = SESSION.get(url, timeout=10)
            logger.info("FMP quote status=%s for %s", getattr(r, "status_code", None), sym)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and len(j) and 'price' in j[0] and j[0].get('price') is not None:
                    price = float(j[0].get('price'))
                    logger.info("Got price %s from FMP for %s", price, sym)
                    return {"price": price, "source": f"FMP (quote:{sym})"}
        except Exception as e:
            logger.info("FMP quote exception for %s: %s", s, str(e))

    # 2) yfinance
    if YFINANCE_AVAILABLE:
        y_candidates = []
        if "." in s:
            y_candidates = [s, s.replace(".NS", ""), s.replace(".BO", "")]
        else:
            y_candidates = [s + ".NS", s, s + ".BO", s + ".NSE"]
        seen = set(); y_candidates = [c for c in y_candidates if not (c in seen or seen.add(c))]
        for cand in y_candidates:
            try:
                logger.info("Trying yfinance %s", cand)
                t = yf.Ticker(cand)
                df = t.history(period="5d", interval="1d")
                if df is not None and not df.empty:
                    last_close = df['Close'].iloc[-1]
                    if last_close:
                        price = float(last_close)
                        logger.info("Got price %s from yfinance %s", price, cand)
                        return {"price": price, "source": f"Yahoo ({cand})"}
                # fallback: ticker.info
                try:
                    info = t.info
                    for key in ("regularMarketPrice", "previousClose", "currentPrice"):
                        if key in info and info.get(key):
                            price = float(info[key])
                            logger.info("Got price %s from yfinance.info[%s] %s", price, key, cand)
                            return {"price": price, "source": f"Yahoo.info({cand})"}
                except Exception:
                    continue
            except Exception as e:
                logger.info("yfinance %s failed: %s", cand, str(e))
    else:
        logger.info("yfinance not available")

    # 3) NSEpy
    try:
        from nsepy import get_history
        end = date.today()
        start = end - timedelta(days=7)
        sym_for_nse = s.split(".")[0]
        df = get_history(symbol=sym_for_nse, start=start, end=end)
        if df is not None and not df.empty:
            last_row = df.reset_index().iloc[-1]
            last_close = last_row.get("Close") or last_row.get("close")
            if last_close:
                price = float(last_close)
                logger.info("Got price %s from NSEpy %s", price, s)
                return {"price": price, "source": "NSEpy"}
    except Exception as e:
        logger.info("NSEpy lookup failed for %s: %s", s, str(e))

    logger.info("No provider returned price for %s", s)
    return None


def fetch_profile(symbol: str, api_key: Optional[str]) -> dict:
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
# NSEpy history
# ------------------------------
def _fetch_history_nsepy(symbol: str, period_years: int = 3) -> List[dict]:
    try:
        from nsepy import get_history
    except Exception as e:
        logger.info("nsepy not available: %s", str(e))
        return []
    s = symbol.strip().upper()
    candidates = [s, s.replace(".NS", ""), s.replace(".BO", "")]
    seen = set(); candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    end = date.today(); start = end - timedelta(days=365 * period_years)
    for cand in candidates:
        try:
            sym_for_nse = cand.split(".")[0]
            df = get_history(symbol=sym_for_nse, start=start, end=end)
            if df is None or df.empty:
                continue
            df = df.reset_index()
            rows = []
            for _, r in df.iterrows():
                dstr = r["Date"].strftime("%Y-%m-%d") if hasattr(r.get("Date"), "strftime") else str(r.get("Date"))
                rows.append({
                    "date": dstr,
                    "open": float(r.get("Open", 0) or 0),
                    "high": float(r.get("High", 0) or 0),
                    "low": float(r.get("Low", 0) or 0),
                    "close": float(r.get("Close", 0) or 0),
                    "volume": int(r.get("Volume", 0) or 0)
                })
            if rows:
                return list(reversed(rows))
        except Exception as ex:
            logger.info("nsepy history failed %s: %s", cand, str(ex))
            continue
    return []


# ------------------------------
# yfinance history
# ------------------------------
def _fetch_history_yfinance(symbol: str, period: str = "2y", interval: str = "1d") -> List[dict]:
    if not YFINANCE_AVAILABLE:
        return []
    s = symbol.strip().upper()
    candidates = [s + ".NS", s, s + ".BO", s + ".NSE"]
    seen = set(); candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    for cand in candidates:
        try:
            t = yf.Ticker(cand)
            df = t.history(period=period, interval=interval)
            if df is None or df.empty:
                continue
            df = df.reset_index()
            rows = []
            for _, r in df.iterrows():
                dstr = r["Date"].strftime("%Y-%m-%d") if hasattr(r.get("Date"), "strftime") else str(r.get("Date"))
                rows.append({
                    "date": dstr,
                    "open": float(r.get('Open', 0) or 0),
                    "high": float(r.get('High', 0) or 0),
                    "low": float(r.get('Low', 0) or 0),
                    "close": float(r.get('Close', 0) or 0),
                    "volume": int(r.get('Volume', 0) or 0)
                })
            if rows:
                return list(reversed(rows))
        except Exception as e:
            logger.info("yfinance history failed %s: %s", cand, str(e))
            continue
    return []


# ------------------------------
# Main fetch_history
# ------------------------------
def fetch_history(symbol: str, api_key: Optional[str], timeseries: int = 180) -> Tuple[List[dict], str]:
    tried = []
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
                        logger.info("Fetched %d rows from FMP for %s (ts=%d)", len(hist), sym, ts)
                        return hist, f"FMP (ts={ts})"
                elif r.status_code == 403:
                    logger.warning("FMP 403 for %s (ts=%d)", sym, ts)
            except Exception as e:
                logger.info("FMP fetch error %s: %s", sym, str(e))
    nse_rows = _fetch_history_nsepy(symbol, 3)
    if nse_rows:
        return nse_rows, "NSEpy"
    yf_rows = _fetch_history_yfinance(symbol, "2y")
    if yf_rows:
        return yf_rows, "Yahoo"
    raise RuntimeError(f"No history for {symbol}. Tried {tried}.")
