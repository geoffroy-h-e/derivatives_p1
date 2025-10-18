# q5_b.py
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from pathlib import Path

# CRR option pricing
def crr_price_stable(S, K, T, r, y, sigma, option_type="C", steps=2000, max_steps=600):
    if T <= 0:
        return max(0.0, (S-K) if option_type=="C" else (K-S))
    N = int(min(max(1, steps), max_steps))
    dt = T / N
    if sigma <= 0:
        f = S*np.exp((r-y)*T)
        return max(0.0, (f - K)*np.exp(-r*T)) if option_type=="C" else max(0.0, (K - f)*np.exp(-r*T))
    u = np.exp(sigma*np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - y) * dt) - d) / (u - d)
    p = float(np.clip(p, 1e-12, 1-1e-12))
    disc = np.exp(-r * dt)
    log_S0 = np.log(S) + N*np.log(d)
    S_k = np.exp(log_S0)
    payoff = np.empty(N+1, dtype=np.float64)
    if option_type == "C":
        for k in range(N+1):
            payoff[k] = max(S_k - K, 0.0)
            S_k *= (u/d)
    else:
        for k in range(N+1):
            payoff[k] = max(K - S_k, 0.0)
            S_k *= (u/d)
    for _ in range(N):
        payoff = disc * (p*payoff[1:] + (1.0 - p)*payoff[:-1])
    return float(payoff[0])

# Compute implied volatility using CRR
def implied_vol_crr_stable(S, K, T, r, y, market_price, option_type, steps, max_steps=600):
    def f(sig):
        return crr_price_stable(S, K, T, r, y, sig, option_type, steps=steps, max_steps=max_steps) - market_price
    try:
        lo, hi = 1e-6, 3.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            for hi_try in (5.0, 8.0, 12.0):
                if f(lo)*f(hi_try) <= 0:
                    hi = hi_try
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan

# Apply CRR IV calculation row by row
def rows_iv_crr(df):
    iv = pd.Series(index=df.index, dtype=float)
    for idx, r in df.iterrows():
        S   = float(r["stock_price"])
        K   = float(r["strike"])
        T   = float(r["YTM"])
        rr  = float(r["risk_free"])
        y   = 0.0 if T <= 0 or r["stock_exdiv"] <= 0 else max(0.0, (1.0/T)*np.log(r["stock_price"]/r["stock_exdiv"]))
        steps = int(max(1, 5 * float(r["DTM"])))
        steps = min(steps, 600)
        cpval = str(r.get("cp_flag","")).upper()
        if not cpval and "is_call" in r.index:
            cpval = "C" if int(r["is_call"])==1 else "P"
        cp   = "C" if cpval.startswith("C") else "P"
        iv.loc[idx] = implied_vol_crr_stable(S, K, T, rr, y, float(r["option_price"]), cp, steps)
    return iv

def _summ(s):
    return s.describe()[["count","mean","std","min","25%","50%","75%","max"]]

def _maybe_display(df, name=None):
    try:
        display(df)
    except Exception:
        if name:
            print(name)
        print(df.to_string())

def _ensure_fields(options):
        if "cp_flag" not in options.columns:
            if "is_call" in options.columns:
                options["cp_flag"] = np.where(options["is_call"]==1,"C","P")
            elif "option_type" in options.columns:
                options["cp_flag"] = options["option_type"].astype(str).str.upper().str[0]
        for c in ["date","exdate"]:
            if c in options.columns:
                options[c] = pd.to_datetime(options[c]).dt.tz_localize(None).dt.normalize()
        return options

# Main analysis and plotting

def run_q5(options):
    options = options.copy()
    options = _ensure_fields(options)
    options["iv_crr"] = rows_iv_crr(options)
    options["moneyness"] = options["strike"] / options["stock_exdiv"]
    options["diff_crr_vs_provider_pct"] = 100 * (options["impl_volatility"] / options["iv_crr"] - 1)
    if "implied_vol_bms" in options.columns:
        options["abs_err_crr_vs_provider"] = (options["iv_crr"] - options["impl_volatility"]).abs()
        options["abs_err_bms_vs_provider"] = (options["implied_vol_bms"] - options["impl_volatility"]).abs()
    else:
        options["abs_err_crr_vs_provider"] = (options["iv_crr"] - options["impl_volatility"]).abs()
        options["abs_err_bms_vs_provider"] = np.nan
    stats_pct = _summ(options["diff_crr_vs_provider_pct"].dropna()).to_frame("CRR vs provider (% diff)")
    if options["abs_err_bms_vs_provider"].notna().any():
        stats_abs = pd.concat([
            _summ(options["abs_err_crr_vs_provider"].dropna()).rename("abs|CRR−provider"),
            _summ(options["abs_err_bms_vs_provider"].dropna()).rename("abs|BMS−provider")
        ], axis=1)
    else:
        stats_abs = _summ(options["abs_err_crr_vs_provider"].dropna()).to_frame("abs|CRR−provider")
    print("Q5 — Summary stats (ALL options)")
    _maybe_display(stats_pct.round(4))
    _maybe_display(stats_abs.round(6))
    if options["abs_err_bms_vs_provider"].notna().any():
        closer_counts = pd.Series(
            np.where(options["abs_err_crr_vs_provider"] < options["abs_err_bms_vs_provider"], "CRR closer",
            np.where(options["abs_err_crr_vs_provider"] > options["abs_err_bms_vs_provider"], "BMS closer", "Tie"))
        ).value_counts(dropna=False)
        print("Count of rows by which model is closer to provider IV:")
        _maybe_display(closer_counts)
    sns.set_context("talk"); sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    palette_cp = {"C": "steelblue", "P": "orange"}
    label_cp    = {"C": "Call (C)", "P": "Put (P)"}
    sub_left = options[options["date"] == pd.Timestamp("2020-01-17")].copy()
    sub_left["diff_clip"] = sub_left["diff_crr_vs_provider_pct"].clip(-50, 50)
    sns.scatterplot(data=sub_left, x="moneyness", y="diff_clip", hue="cp_flag", palette=palette_cp, alpha=0.6, s=25, ax=axes[0])
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Jan 17, 2020")
    axes[0].set_xlabel("Moneyness (K / S_exdiv)")
    axes[0].set_ylabel("Provider vs CRR IV (% diff)")
    leg = axes[0].get_legend()
    if leg: leg.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [label_cp.get(l, l) for l in labels]
    axes[0].legend(handles, labels, title="Option Type", loc="upper right", frameon=True, fontsize=9, title_fontsize=10)
    sub_right = options[options["date"] == pd.Timestamp("2020-03-20")].copy()
    sub_right["diff_clip"] = sub_right["diff_crr_vs_provider_pct"].clip(-50, 50)
    sns.scatterplot(data=sub_right, x="moneyness", y="diff_clip", hue="cp_flag", palette=palette_cp, alpha=0.6, s=25, ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Mar 20, 2020")
    axes[1].set_xlabel("Moneyness (K / S_exdiv)")
    axes[1].set_ylabel("")
    leg = axes[1].get_legend()
    if leg: leg.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    labels = [label_cp.get(l, l) for l in labels]
    axes[1].legend(handles, labels, title="Option Type", loc="upper right", frameon=True, fontsize=9, title_fontsize=10)
    fig.suptitle("Provider IV vs CRR IV with Dividend Yield — ALL options (no OTM filter)")
    plt.tight_layout()
    plt.show()
    if options["abs_err_bms_vs_provider"].notna().any():
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        comp = pd.DataFrame({
            "abs|CRR−provider": options["abs_err_crr_vs_provider"],
            "abs|BMS−provider": options["abs_err_bms_vs_provider"]
        }).melt(var_name="Model", value_name="Abs Error")
        sns.boxplot(data=comp.dropna(), x="Model", y="Abs Error", ax=ax2)
        ax2.set_title("Which model is closer to provider IV? (ALL options)")
        ax2.set_xlabel("")
        ax2.set_ylabel("Absolute error vs provider IV")
        plt.tight_layout()
        plt.show()
    return options

def _load_options_from_path(p):
    p = Path(p)
    if p.suffix.lower() in [".parquet",".pq",".parq"]:
        return pd.read_parquet(p)
    if p.suffix.lower() in [".csv",".txt"]:
        return pd.read_csv(p)
    if p.suffix.lower() in [".pkl",".pickle"]:
        return pd.read_pickle(p)
    raise ValueError("unsupported file type")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        opts_df = _load_options_from_path(args[0])
        run_q5(opts_df)
    else:
        for cand in ["options.parquet","options.csv","options.pkl"]:
            if Path(cand).exists():
                run_q5(_load_options_from_path(cand))
                break
        else:
            raise SystemExit("provide options DataFrame via file path or import run_q5 and pass a DataFrame")
