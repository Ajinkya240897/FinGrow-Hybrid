import time, math
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from cachetools import TTLCache
from .utils import fetch_current_price, fetch_history, fetch_profile

# in-memory cache
CACHE = TTLCache(maxsize=200, ttl=60*60*4)
HORIZON_DAYS = {'3-15_days':10, '1-3_months':45, '3-6_months':120, '1-3_years':540}

def build_features_from_history(hist, lookback=30, forward=30):
    if not hist or len(hist) < (lookback + forward):
        return None
    hist_old = list(reversed(hist))
    closes = [float(h['close']) for h in hist_old]
    df = pd.DataFrame({'close': closes})
    df['ret1'] = df['close'].pct_change().fillna(0)
    df['ret5'] = df['close'].pct_change(5).fillna(0)
    df['sma10'] = df['close'].rolling(10).mean().fillna(method='bfill')
    df['sma30'] = df['close'].rolling(30).mean().fillna(method='bfill')
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    df['rsi'] = 100 - 100/(1+rs)
    rows = []
    for i in range(0, len(df) - lookback - forward + 1):
        win = df.iloc[i:i+lookback]
        future = df.iloc[i+lookback+forward-1]
        feat = [
            win['close'].iloc[-1],
            win['ret1'].iloc[-1],
            win['ret5'].iloc[-1],
            (win['sma10'].iloc[-1] - win['sma30'].iloc[-1]),
            win['macd'].iloc[-1],
            win['rsi'].iloc[-1],
            win['close'].std()
        ]
        label = (future['close'] - win['close'].iloc[-1]) / win['close'].iloc[-1]
        rows.append((feat, label))
    if not rows:
        return None
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y

def indicator_only_recommendation(current_price, features_last):
    rsi = features_last.get('rsi', 50)
    sma_diff = features_last.get('sma_diff', 0)
    if rsi > 70:
        return 'Sell', 'RSI is high which often precedes pullbacks.'
    if sma_diff > 0:
        return 'Buy', 'Price is above shorter-term average which indicates upward trend.'
    return 'Hold', 'No strong signals found — consider waiting.'

def build_features_last(hist):
    closes = [float(h['close']) for h in reversed(hist)]
    if len(closes) < 5:
        return {}
    s10 = pd.Series(closes).rolling(10).mean().iloc[-1]
    s30 = pd.Series(closes).rolling(30).mean().iloc[-1]
    rsi_val = pd.Series(closes).diff().clip(lower=0).rolling(14).mean().iloc[-1] if len(closes) >= 14 else 50
    return {'sma10': s10, 'sma30': s30, 'sma_diff': s10 - s30, 'rsi': float(rsi_val)}

def predict_stock(ticker, horizon, fmp_key):
    key = fmp_key
    cache_key = f"{ticker}:{horizon}:{key}"
    if cache_key in CACHE:
        return CACHE[cache_key]
    hist = fetch_history(ticker, key, timeseries=800)
    cur = fetch_current_price(ticker, key)
    profile = fetch_profile(ticker, key) if key else {}
    if not hist or not cur:
        raise RuntimeError("Unable to fetch data for ticker. Check FMP API key and symbol.")
    features_last = build_features_last(hist)
    lf = 30
    forward = {'3-15_days':10, '1-3_months':45, '3-6_months':120, '1-3_years':540}.get(horizon,45)
    XY = build_features_from_history(hist, lookback=lf, forward=forward)
    if XY is None:
        decision, reason = indicator_only_recommendation(cur['price'], features_last)
        out = {
            "ticker": ticker,
            "current_price": round(float(cur['price']),2),
            "predicted_price": round(float(cur['price']),2),
            "predicted_return_percent": 0.0,
            "momentum": "Neutral",
            "fundamentals_score": 50,
            "recommendation": {
                "decision": decision,
                "buy_below": round(float(cur['price']) * 0.95, 2),
                "sell_target": round(float(cur['price']) * 1.03, 2),
                "stop_loss": round(float(cur['price']) * 0.9, 2),
                "plain_language_reason": reason
            }
        }
        CACHE[cache_key] = out
        return out
    X, y = XY
    try:
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        resid = y - preds
        resid_std = float(np.std(resid))
        last_feats = X[-1].reshape(1, -1)
        pred_return = float(model.predict(last_feats)[0])
    except Exception:
        try:
            rf = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42)
            rf.fit(X, y)
            pred_return = float(rf.predict(X[-1].reshape(1,-1))[0])
            resid_std = float(np.std(y - rf.predict(X)))
        except Exception:
            decision, reason = indicator_only_recommendation(cur['price'], features_last)
            out = {
                "ticker": ticker,
                "current_price": round(float(cur['price']),2),
                "predicted_price": round(float(cur['price']),2),
                "predicted_return_percent": 0.0,
                "momentum": "Neutral",
                "fundamentals_score": 50,
                "recommendation": {
                    "decision": decision,
                    "buy_below": round(float(cur['price']) * 0.95, 2),
                    "sell_target": round(float(cur['price']) * 1.03, 2),
                    "stop_loss": round(float(cur['price']) * 0.9, 2),
                    "plain_language_reason": reason
                }
            }
            CACHE[cache_key] = out
            return out
    current_price = float(cur['price'])
    days = forward
    scale = (days / 30.0) ** 0.5
    predicted_price = current_price * (1 + pred_return * scale)
    implied_return = (predicted_price - current_price) / current_price * 100.0
    spread = max(0.02, abs(pred_return)*0.6 + resid_std * math.sqrt(days/30.0))
    p05 = predicted_price * (1 - spread)
    p95 = predicted_price * (1 + spread)
    momentum = "Positive" if X[-1][1] > 0.005 else ("Negative" if X[-1][1] < -0.005 else "Neutral")
    fscore = 50
    try:
        pe = float(profile.get('priceEarningsRatio') or 0)
        mcap = float(profile.get('mktCap') or 0)
        if pe and pe < 15: fscore += 10
        if mcap and mcap > 1e10: fscore += 10
    except Exception:
        pass
    sigma_q = (p95 - p05)/2
    buy_threshold = predicted_price - 0.8 * sigma_q
    sell_target = predicted_price + 0.5 * sigma_q
    stop_loss = current_price * 0.88
    if current_price <= buy_threshold and implied_return >= 6:
        decision = 'Buy below'
        reason = f"The model expects a good upside ({implied_return:.1f}%) and momentum is {momentum}."
    elif implied_return >= 3 and fscore >= 40:
        decision = 'Hold'
        reason = f"The expected return is {implied_return:.1f}% and fundamentals look reasonable (score {fscore})."
    else:
        decision = 'Sell'
        reason = f"Expected return only {implied_return:.1f}%, momentum {momentum}."
    plain = (
        f"We see current price {current_price:.2f}. The hybrid model predicts ~{predicted_price:.2f} "
        f"for the chosen horizon (implied return {implied_return:.1f}%). {reason} "
        f"Key factors: recent returns, MA/EMA signals and RSI."
    )
    out = {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_return_percent": round(implied_return, 2),
        "confidence_score": max(20, min(95, int(80 - abs(pred_return)*200))),
        "quantiles": {"p05": round(p05,2), "p95": round(p95,2)},
        "momentum": {"label": momentum},
        "fundamentals_score": int(max(1,min(100,fscore))),
        "recommendation": {
            "decision": decision,
            "buy_below": round(buy_threshold,2),
            "sell_target": round(sell_target,2),
            "hold_until": round(sell_target,2),
            "stop_loss": round(stop_loss,2),
            "plain_language_reason": plain
        },
        "top_factors": [
            {"name":"Recent returns","why":"Short-term return pattern"},
            {"name":"MA/EMA/RSI","why":"Technical indicators influence the model"}
        ],
        "model": {"name":"hybrid-on-demand","version":"1.0"}
    }
    CACHE[cache_key] = out
    return out
