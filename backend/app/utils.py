import requests
FMP_QUOTE_URL = "https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={key}"
FMP_HIST_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries={timeseries}&apikey={key}"
FMP_PROFILE = "https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={key}"
COMMON_SUFFIXES = ["", ".NS", ".BO"]

def resolve_symbol(symbol: str, api_key: str):
    if not api_key or not symbol: return symbol
    s = symbol.strip().upper()
    for suf in COMMON_SUFFIXES:
        cand = s if (suf == "" and "." in s) else s + suf
        try:
            r = requests.get(FMP_QUOTE_URL.format(symbol=cand, key=api_key), timeout=6)
            if r.ok:
                j = r.json()
                if isinstance(j, list) and len(j) and 'price' in j[0]:
                    return cand
        except Exception:
            pass
    return s

def fetch_current_price(symbol: str, api_key: str):
    if not api_key: return None
    sym = resolve_symbol(symbol, api_key)
    r = requests.get(FMP_QUOTE_URL.format(symbol=sym, key=api_key), timeout=8)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return {"price": data[0].get("price")}
    return None

def fetch_history(symbol: str, api_key: str, timeseries: int = 800):
    if not api_key: return None
    sym = resolve_symbol(symbol, api_key)
    r = requests.get(FMP_HIST_URL.format(symbol=sym, timeseries=timeseries, key=api_key), timeout=20)
    r.raise_for_status()
    return r.json().get("historical", [])

def fetch_profile(symbol: str, api_key: str):
    if not api_key: return {}
    try:
        r = requests.get(FMP_PROFILE.format(symbol=symbol, key=api_key), timeout=8)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and j: return j[0]
    except Exception:
        pass
    return {}
