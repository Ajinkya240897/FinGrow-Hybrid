# Add at top of file near other imports:
import requests, time
# For fallback:
import datetime
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries={timeseries}&apikey={key}"

def fetch_history(symbol: str, api_key: str, timeseries: int = 800):
    """
    Fetch historical OHLCV list of dicts for a symbol.
    Strategy:
      1) Try FMP with descending timeseries sizes (800,365,180,90,30).
      2) If FMP returns 403 or fails for large series, try smaller sizes.
      3) If still fails, fallback to yfinance if available.
      4) Raise RuntimeError with a friendly message if all fail.
    Returns:
      list of historical rows (each row is a dict with keys like 'date','open','close',...)
    """
    if not api_key:
        # If no API key passed, skip FMP and try yfinance directly
        if YFINANCE_AVAILABLE:
            return _fetch_history_yfinance(symbol)
        raise RuntimeError("No FMP API key provided. Please set FMP_API_KEY in Render environment or provide a key in the UI.")
    sym = resolve_symbol(symbol, api_key)

    # Try descending FMP timeseries sizes (smaller sizes are less likely to be blocked on free plans)
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
                    return hist
                # if response OK but no historical data, continue with smaller ts
                time.sleep(0.12)
                continue
            elif r.status_code == 403:
                # Permission / plan blocked for this timeseries size -> try smaller
                time.sleep(0.12)
                continue
            else:
                # Other 4xx/5xx: raise to be handled by caller
                r.raise_for_status()
        except requests.RequestException:
            # network/work issue, try next smaller ts
            time.sleep(0.12)
            continue

    # If FMP attempts exhausted, fallback to yfinance if available
    if YFINANCE_AVAILABLE:
        try:
            hist_y = _fetch_history_yfinance(symbol)
            if hist_y:
                return hist_y
        except Exception:
            pass

    # If all failed, raise a clear error message for the frontend
    raise RuntimeError(
        f"Unable to fetch historical data for {symbol}. Tried FMP timeseries: {tried}. "
        "This often means your FMP key/plan doesn't allow large history. "
        "If you have a paid plan, ensure the key is correct. Otherwise use a different data source."
    )

def _fetch_history_yfinance(symbol: str, period: str = "1y"):
    """
    Fetch history from yfinance as fallback.
    Returns list of dicts like FinancialModelingPrep returns (date, open, high, low, close, volume).
    """
    # Resolve a ticker for Yahoo: convert common NSE tickers like TCS or RELIANCE to TCS.NS if not already
    s = symbol.strip().upper()
    if "." not in s:
        # try NSE suffix by default for Indian tickers
        ysym = s + ".NS"
    else:
        ysym = s
    # Use yfinance to download
    df = yf.Ticker(ysym).history(period=period, auto_adjust=False)
    if df is None or df.empty:
        # Try without suffix
        df2 = yf.Ticker(s).history(period=period, auto_adjust=False)
        df = df2
        if df is None or df.empty:
            return []
    # Convert dataframe to list of dicts (most recent first) like FMP
    df = df.reset_index()
    rows = []
    for _, r in df.iterrows():
        # Ensure date formatted like 'YYYY-MM-DD'
        date = r['Date'].strftime("%Y-%m-%d") if hasattr(r['Date'], 'strftime') else str(r['Date'])
        rows.append({
            "date": date,
            "open": float(r.get('Open', 0) or 0),
            "high": float(r.get('High', 0) or 0),
            "low": float(r.get('Low', 0) or 0),
            "close": float(r.get('Close', 0) or 0),
            "volume": int(r.get('Volume', 0) or 0)
        })
    # yfinance returns oldest -> newest; we want the same as FMP (which often returns newest first),
    # but other parts of predictor reverse/handle ordering; returning newest-first matches the prior code.
    return list(reversed(rows))  # newest-first
