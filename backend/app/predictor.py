# backend/app/predictor.py
import os
import math
import logging
from typing import Optional
from cachetools import TTLCache, cached
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .utils import fetch_history, fetch_current_price, fetch_profile

logger = logging.getLogger("fingrow.predictor")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Simple in-memory cache for predictions (personal use). TTL = 4 hours
PRED_CACHE = TTLCache(maxsize=2000, ttl=60 * 60 * 4)

def _compute_features_from_df(df: pd.DataFrame):
    """
    df is newest-first. Return X (n_samples, n_features) and y (next_return).
    We'll build features as recent lag returns and rolling volatility.
    """
    if df is None or df.empty:
        return None, None
    # use oldest-first for sliding windows
    df2 = df.sort_values("date", ascending=True).reset_index(drop=True)
    closes = df2["close"].values
    if len(closes) < 8:
        return None, None

    window = 5  # features lookback
    X = []
    y = []
    for i in range(window, len(closes)-1):
        past = closes[i-window:i]
        # log returns
        returns = np.diff(past) / past[:-1]
        feat = list(returns)  # length window-1
        feat += [
            np.mean(returns),
            np.std(returns),
            (closes[i] - np.mean(past)) / np.mean(past)  # price relative to mean
        ]
        X.append(feat)
        # target is next period return
        next_ret = (closes[i+1] - closes[i]) / closes[i]
        y.append(next_ret)
    if not X:
        return None, None
    X = np.array(X)
    y = np.array(y)
    return X, y

def _train_quick_model(X, y):
    # small pipeline: standardize + ridge
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(Xs, y)
    return model, scaler

def _safe_round(v, nd=2):
    try:
        return round(float(v), nd)
    except Exception:
        return None

def _recommendation_from_pred(current_price, predicted_price, confidence):
    # simple rules for recommendation
    if current_price <= 0 or predicted_price <= 0:
        return {"decision": "No Data", "plain_language_reason": "Price data missing."}
    ret_pct = (predicted_price - current_price) / current_price * 100
    if confidence >= 65 and ret_pct >= 5:
        decision = "Strong Buy"
    elif confidence >= 50 and ret_pct >= 3:
        decision = "Buy"
    elif confidence >= 40 and abs(ret_pct) < 3:
        decision = "Hold"
    elif ret_pct <= -5:
        decision = "Strong Sell"
    else:
        decision = "Sell" if ret_pct < 0 else "Hold"
    reason = f"Predicted return {ret_pct:.2f}% with confidence {confidence}/100."
    return {
        "decision": decision,
        "buy_below": _safe_round(current_price * 0.98, 2),
        "sell_target": _safe_round(predicted_price, 2),
        "hold_until": _safe_round(predicted_price, 2),
        "stop_loss": _safe_round(current_price * 0.9, 2),
        "plain_language_reason": reason
    }

@cached(PRED_CACHE)
def predict_stock(ticker: str, horizon: str, fmp_key: Optional[str]):
    """
    Main predict function.
    Returns a dict with keys:
      ticker, current_price, predicted_price, predicted_return_percent,
      confidence_score, quantiles, momentum, fundamentals_score, recommendation, top_factors, model
    """
    ticker = ticker.strip().upper()
    # 1) fetch current price
    current = fetch_current_price(ticker, fmp_key) or {"price": None}
    current_price = float(current.get("price") or 0.0)

    # 2) try history
    try:
        rows, hist_source = fetch_history(ticker, fmp_key, timeseries=180)
    except Exception as e:
        logger.info("History fetch failed for %s: %s", ticker, str(e))
        # fallback conservative estimate (low-confidence)
        horizon_map = {
            '3-15_days': 0.01,
            '1-3_months': 0.03,
            '3-6_months': 0.06,
            '1-3_years': 0.12
        }
        drift = horizon_map.get(horizon, 0.03)
        predicted_price = _safe_round(current_price * (1 + drift), 2) if current_price > 0 else 0.0
        implied_return = _safe_round(((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0.0, 2)
        profile = fetch_profile(ticker, fmp_key) or {}
        out = {
            "ticker": ticker,
            "current_price": _safe_round(current_price,2),
            "predicted_price": predicted_price,
            "predicted_return_percent": implied_return,
            "confidence_score": 18,
            "quantiles": {"p05": _safe_round(predicted_price * 0.9,2), "p95": _safe_round(predicted_price * 1.1,2)},
            "momentum": {"label": "Neutral"},
            "fundamentals_score": int(profile.get("rating", 50) if isinstance(profile.get("rating", None), (int, float)) else 50),
            "recommendation": {
                "decision": "Hold",
                "buy_below": _safe_round(current_price * 0.97,2),
                "sell_target": predicted_price,
                "hold_until": predicted_price,
                "stop_loss": _safe_round(current_price * 0.9,2),
                "plain_language_reason": (
                    "We could not fetch historical data for this ticker from the data sources. "
                    "So we returned a conservative, low-confidence estimate using the current price and a short-term rule-of-thumb. "
                    "Provide a paid data API key or try again later for a full prediction."
                )
            },
            "top_factors": [
                {"name": "Data status", "why": "Checked multiple providers; no usable history"},
                {"name": "Fallback logic", "why": f"Used fallback drift {drift*100:.1f}% for horizon {horizon}"}
            ],
            "model": {"name":"fallback-indicator","version":"0.1"},
            "data_source": "None"
        }
        return out

    # Build DataFrame from rows (rows are newest-first)
    df = None
    if rows:
        try:
            df = pd.DataFrame(rows)
            # ensure numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            # newest-first is already expected by code; predictor will convert where needed
        except Exception as e:
            logger.info("Error building dataframe for %s: %s", ticker, str(e))
            df = None

    n_hist = len(rows) if rows else 0

    # Compute momentum simple label using last 5 close returns if available
    momentum_label = "Neutral"
    if df is not None and len(df) >= 6:
        closes = list(df['close'])
        # newest-first -> last returns use first elements
        try:
            last5 = closes[:6]  # newest-first slice
            # compute percent change over last 5 days
            pct = (last5[0] - last5[-1]) / (last5[-1] or 1)
            if pct > 0.06:
                momentum_label = "Strong Positive"
            elif pct > 0.015:
                momentum_label = "Positive"
            elif pct < -0.06:
                momentum_label = "Strong Negative"
            elif pct < -0.015:
                momentum_label = "Negative"
            else:
                momentum_label = "Neutral"
        except Exception:
            momentum_label = "Neutral"

    # 3) Try to use pre-trained model artifact if present
    model_path = os.path.join("models", f"{ticker}.joblib")
    predicted_price = None
    model_used = None
    residual_std = 0.05  # default residual for quantiles

    if os.path.exists(model_path):
        try:
            mobj = joblib.load(model_path)
            # mobj expected as dict {'model': model, 'scaler': scaler, 'features': features_meta}
            model = mobj.get("model")
            scaler = mobj.get("scaler")
            # build a feature vector from most recent data if possible
            Xfeat, _ = _compute_features_from_df(df)
            if Xfeat is not None and Xfeat.shape[0] >= 1:
                last_feat = Xfeat[-1].reshape(1, -1)
                if scaler is not None:
                    last_feat = scaler.transform(last_feat)
                pred_ret = model.predict(last_feat)[0]
                predicted_price = _safe_round(current_price * (1 + float(pred_ret)), 2)
                model_used = f"pretrained:{os.path.basename(model_path)}"
                # compute residual std from training metadata if available
                residual_std = float(mobj.get("resid_std", residual_std))
        except Exception as e:
            logger.info("Failed loading pretrained model for %s: %s", ticker, str(e))
            model_used = None

    # 4) If no pretrained model, but sufficient history, train quick ridge
    if predicted_price is None and df is not None and n_hist >= 40:
        X, y = _compute_features_from_df(df)
        if X is not None and len(y) >= 10:
            try:
                model, scaler = _train_quick_model(X, y)
                # predict using last row
                lastX = scaler.transform(X[-1].reshape(1, -1))
                pred_ret = model.predict(lastX)[0]
                predicted_price = _safe_round(current_price * (1 + float(pred_ret)), 2)
                model_used = "quick-ridge"
                residual_std = float(np.std(y - model.predict(scaler.transform(X))))
            except Exception as e:
                logger.info("Quick model train failed for %s: %s", ticker, str(e))
                predicted_price = None

    # 5) If still no predicted_price, fallback to simple drift based on horizon
    if predicted_price is None:
        horizon_map = {
            '3-15_days': 0.01,
            '1-3_months': 0.03,
            '3-6_months': 0.06,
            '1-3_years': 0.12
        }
        drift = horizon_map.get(horizon, 0.03)
        predicted_price = _safe_round(current_price * (1 + drift), 2) if current_price > 0 else 0.0
        model_used = model_used or "fallback-drfit"
        residual_std = max(residual_std, 0.06)

    # quantiles using residual_std (conservative)
    p05 = _safe_round(predicted_price * (1 - 2 * residual_std), 2) if predicted_price else None
    p95 = _safe_round(predicted_price * (1 + 2 * residual_std), 2) if predicted_price else None

    implied_return = _safe_round(((predicted_price - current_price) / current_price * 100) if current_price > 0 else 0.0, 2)

    # fundamentals score heuristic
    profile = fetch_profile(ticker, fmp_key) or {}
    try:
        fundamentals_score = int(max(1, min(100, int(profile.get("rating") or profile.get("score") or 50))))
    except Exception:
        fundamentals_score = 50

    # Compute base confidence and adjust by source + history length
    base_conf = 60
    src = hist_source or "Unknown"
    if src.startswith("NSEpy"):
        source_bonus = 15
    elif src.startswith("FMP"):
        source_bonus = 12
    elif src.lower().startswith("yahoo") or src.lower().startswith("yfinance"):
        source_bonus = 5
    else:
        source_bonus = 0
    length_bonus = min(30, int(n_hist / 10))
    penalty = int(min(30, abs(implied_return) * 2))
    final_conf = max(5, min(95, int(base_conf + source_bonus + length_bonus - penalty)))

    # Build recommendation
    recommendation = _recommendation_from_pred(current_price, predicted_price, final_conf)

    out = {
        "ticker": ticker,
        "current_price": _safe_round(current_price, 2),
        "predicted_price": predicted_price,
        "predicted_return_percent": implied_return,
        "confidence_score": final_conf,
        "quantiles": {"p05": p05, "p95": p95},
        "momentum": {"label": momentum_label},
        "fundamentals_score": fundamentals_score,
        "recommendation": recommendation,
        "top_factors": [
            {"name": "Data source", "why": f"{src} (n={n_hist})"},
            {"name": "Model used", "why": model_used or "fallback"},
            {"name": "Recent momentum", "why": momentum_label}
        ],
        "model": {"name": model_used or "fallback", "version": "1.0"},
        "data_source": f"{src} (n={n_hist})"
    }

    logger.info("PREDICT | ticker=%s source=%s n=%d conf=%d model=%s", ticker, src, n_hist, final_conf, model_used)
    return out
