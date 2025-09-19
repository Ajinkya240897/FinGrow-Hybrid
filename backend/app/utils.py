# backend/app/utils.py
import requests
import time
import datetime

# Fallback to yfinance if present
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# FMP endpoints
FMP_QUOTE_URL = "https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={key}"
FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries={timeseries}&apikey={key}"
FMP_PROFILE = "https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={key}"

# Common suffixes to try for Indian tickers
COMMON_SUFFIXES = ["", ".NS", ".BO"]

def resolve_symbol(symbol: str, api_key: str):
    """
    Try candidate suffixes and return the first symbol accepted by FMP quote endpoint.
    If no symbol is validated, return original uppercase symbol.
    """
    if not api_key or not symbol:
        return symbol.strip().upper()
    s = symbol.strip().upper()
    for suf in COMMON_SUFFIXES:
        candidate = s if (suf == "" and "." in s) else s + suf
        try:
            url = FMP_QUOTE_URL.format(symbol=candidate, key=api_key)
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and len(j) and 'price' in j[0]:
                    return candidate
        except Exception:
            # ignore and try next suffix
            pass
    # fallback to input symbol uppercased
    return s

def fetch_current_price(symbol: str, api_key: str):
    """
    Fetch current quote (short) from FMP for resolved symbol.
    Returns dict like {"price": <float>} or None on failure.
    """
    if not api_key:
        return None
    resolved = resolve_symbol(symbol, api_key)
    try:
        url = FMP_QUOTE_URL.format(symbol=resolved, key=api_key)
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return {"price": data[0].get('price')}
    except Exception:
        return None
    return None

def fetch_profile(symbol: str, api_key: str):
    """
    Fetch company profile from FMP (used for a simple fundamentals score).
    Returns dict or {}.
    """
    if not api_key:
        return {}
    try:
        url = FMP_PROFILE.format(symbol=symbol, key=api_key)
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j:
            return j[0]
    except Exception:
        pass
    return {}

# ---------------------------
# New tolerant fetch_history
# ---------------------------
def fetch_history(symbol: str, api_key: str, timeseries: int = 800):
    """
    Fetch historical OHLCV rows for a symbol.
    Strategy:
      1) Try FMP with descending timeseries sizes (800, 365, 180, 90, 30).
      2) If FMP returns 403 or otherwise fails for large series, try smaller sizes.
      3) If FMP attempts fail, fallback to yfinance (if available).
      4) Raise RuntimeError with a friendly message if all fail.
    Returns:
      list of historical rows (each a dict with keys like 'date','open','high','low','close','volume'),
      newest-first (matching previous FMP usage).
    """
    # If no API key provided, skip FMP and try yfinance directly (if available)
    if not api_key:
        if YFINANCE_AVAILABLE:
            return _fetch_history_yfinance(symbol)
        raise RuntimeError("No FMP API key provided. Please set FMP_API_KEY in Render environment or provide a key in the UI.")

    # First try FMP with a few descending sizes
    sym = resolve_symbol(symbol, api_key)
    tried = []
    for ts in (timeseries, 365, 180, 90, 30):
        tried.append(ts)
        try:
            url = FMP_HIST_URL.format(symbol=sym, timeseries=ts, key=api_key)
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                hist = data.get("historical") or []
                if hist:
                    # FMP returns newest-first already in many responses; return as-is
                    return hist
                # if no historical inside, try smaller ts
                time.sleep(0.12)
                continue
            elif r.status_code == 403:
                # Likely a permission / plan limitation — try smaller timeseries
                time.sleep(0.12)
                continue
            else:
                # Other 4xx/5xx — bubble up helpful info
                r.raise_for_status()
        except requests.RequestException:
            # network or temporary error — try next smaller ts
            time.sleep(0.12)
            continue

    # If FMP exhausted, try yfinance fallback
    if YFINANCE_AVAILABLE:
        try:
            hist_y = _fetch_history_yfinance(symbol)
            if hist_y:
                return hist_y
        except Exception:
            pass

    # All sources exhausted
    raise RuntimeError(
        f"Unable to fetch historical data for {symbol}. Tried FMP timeseries: {tried}. "
        "This usually means your FMP API key or plan doesn't allow large historical endpoints or you have hit rate limits. "
        "Please check your FMP plan/key or provide a user API key in the UI."
    )

def _fetch_history_yfinance(symbol: str, period: str = "1y"):
    """
    Fetch history from yfinance as fallback and convert to list-of-dicts similar to FMP format.
    Returns newest-first list of dicts with keys: date, open, high, low, close, volume.
    """
    s = symbol.strip().upper()
    # Heuristic: if no dot, assume NSE ticker (append .NS)
    if "." not in s:
        ysym = s + ".NS"
    else:
        ysym = s

    # Prefer the suffixed ticker first; if no data, try unsuffixed
    try:
        t = yf.Ticker(ysym)
        df = t.history(period=period, auto_adjust=False)
        if df is None or df.empty:
            # try without suffix
            t2 = yf.Ticker(s)
            df = t2.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                return []
    except Exception:
        # fallback attempt without suffix
        try:
            t2 = yf.Ticker(s)
            df = t2.history(period=period, auto_adjust=False)
            if df is None or df.empty:
                return []
        except Exception:
            return []

    # Convert dataframe to list of dicts
    df = df.reset_index()
    rows = []
    for _, r in df.iterrows():
        # Date may be a Timestamp; format as YYYY-MM-DD
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
    # yfinance returns oldest-first; convert to newest-first to match FMP-style where used earlier
    return list(reversed(rows))
