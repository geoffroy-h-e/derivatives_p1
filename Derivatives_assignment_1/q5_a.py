from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import brentq

def crr_price_stable(S, K, T, r, y, sigma, option_type="C", steps=2000, max_steps=600):
    if T <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    N = int(min(max(1, steps), max_steps))
    dt = T / N
    if sigma <= 0:
        f = S * np.exp((r - y) * T)
        return max(0.0, (f - K) * np.exp(-r * T)) if option_type == "C" else max(0.0, (K - f) * np.exp(-r * T))
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (np.exp((r - y) * dt) - d) / (u - d)
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    disc = float(np.exp(-r * dt))
    log_S0 = np.log(S) + N * np.log(d)
    S_k = np.exp(log_S0)
    payoff = np.empty(N + 1, dtype=np.float64)
    if option_type == "C":
        for k in range(N + 1):
            payoff[k] = max(S_k - K, 0.0)
            S_k *= (u / d)
    else:
        for k in range(N + 1):
            payoff[k] = max(K - S_k, 0.0)
            S_k *= (u / d)
    for _ in range(N):
        payoff = disc * (p * payoff[1:] + (1.0 - p) * payoff[:-1])
    return float(payoff[0])

def implied_vol_crr_stable(S, K, T, r, y, market_price, option_type, steps, max_steps=600):
    def f(sig):
        return crr_price_stable(S, K, T, r, y, sig, option_type, steps=steps, max_steps=max_steps) - market_price
    try:
        lo, hi = 1e-6, 3.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            for hi_try in (5.0, 8.0, 12.0):
                if f(lo) * f(hi_try) <= 0:
                    hi = hi_try
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan

def rows_iv_crr(df: pd.DataFrame) -> pd.Series:
    iv = pd.Series(index=df.index, dtype=float)
    for idx, r in df.iterrows():
        S   = float(r["stock_price"])
        K   = float(r["strike"])
        T   = float(r["YTM"])
        rr  = float(r["risk_free"])
        y   = 0.0 if T <= 0 or r["stock_exdiv"] <= 0 else max(0.0, (1.0/T)*np.log(r["stock_price"]/r["stock_exdiv"]))
        steps = int(max(1, 5 * float(r["DTM"])))
        steps = min(steps, 600)
        cp   = "C" if str(r["cp_flag"]).upper().startswith("C") else "P"
        iv.loc[idx] = implied_vol_crr_stable(S, K, T, rr, y, float(r["option_price"]), cp, steps)
    return iv

def build_crr_iv_diff_df(options: pd.DataFrame) -> pd.DataFrame:
    opts = options.copy()
    if "iv_crr" not in opts.columns:
        opts["iv_crr"] = rows_iv_crr(opts)
    valid = opts[["strike","stock_exdiv","impl_volatility","iv_crr"]].dropna().copy()
    valid["moneyness"] = valid["strike"] / valid["stock_exdiv"]
    valid["iv_diff_pct"] = 100 * (valid["impl_volatility"] / valid["iv_crr"] - 1)
    return valid

if __name__ == "__main__":
    if "options" not in globals():
        raise RuntimeError("Please define `options` DataFrame in session before running q5_a.py.")
    options = options.copy()
    options["iv_crr"] = rows_iv_crr(options)
    valid = build_crr_iv_diff_df(options)
    print(valid.head())
