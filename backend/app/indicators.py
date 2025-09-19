import numpy as np
import pandas as pd

def sma(arr, n):
    return pd.Series(arr).rolling(n).mean().to_list()

def ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().to_list()

def rsi(arr, n=14):
    s = pd.Series(arr)
    delta = s.diff().dropna()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-9)
    return (100 - 100/(1+rs)).to_list()

def macd(arr, fast=12, slow=26, signal=9):
    efast = pd.Series(arr).ewm(span=fast, adjust=False).mean()
    eslow = pd.Series(arr).ewm(span=slow, adjust=False).mean()
    macd_line = efast - eslow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.to_list(), signal_line.to_list(), hist.to_list()
