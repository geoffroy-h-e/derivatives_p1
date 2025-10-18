# q6_early_ex_premium.py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm

# Black–Scholes price
def bsm_price(S, K, T, r, q, sigma, cp_flag):
    if T <= 0:
        return max(0.0, (S - K) if cp_flag == "C" else (K - S))
    if sigma is None or np.isnan(sigma) or sigma <= 0:
        F = S * np.exp((r - q) * T)
        intrinsic_fwd = max(0.0, F - K) if cp_flag == "C" else max(0.0, K - F)
        return np.exp(-r * T) * intrinsic_fwd
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if cp_flag == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

 #dd dividend yield
def add_div_yield(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    has_S = "stock_price" in d.columns
    S  = d["stock_price"].astype(float) if has_S else d["stock_exdiv"].astype(float)
    Sx = d["stock_exdiv"].astype(float)
    T  = d["YTM"].astype(float)
    y = np.where(has_S & (T > 0) & (Sx > 0) & (S > 0), np.log(S / Sx) / T, 0.0)
    d["div_yield"] = np.maximum(y, 0.0)
    d["S_used"] = S if has_S else Sx
    d["q_used"] = d["div_yield"] if has_S else 0.0
    return d

# Compute European equivalent price under BSM
def compute_euro_equivalent(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "cp_flag" not in d.columns:
        if "is_call" in d.columns:
            d["cp_flag"] = np.where(d["is_call"]==1,"C","P")
        elif "option_type" in d.columns:
            d["cp_flag"] = d["option_type"].astype(str).str.upper().str[0]
    d["cp_flag"] = d["cp_flag"].astype(str).str.upper().str[0]
    sig = d["impl_volatility"].astype(float)
    d["euro_equiv"] = [
        bsm_price(S=float(S), K=float(K), T=float(T), r=float(r), q=float(q),
                  sigma=float(s) if pd.notna(s) else np.nan, cp_flag=cp)
        for S, K, T, r, q, s, cp in zip(
            d["S_used"], d["strike"], d["YTM"], d["risk_free"], d["q_used"],
            sig, d["cp_flag"]
        )
    ]
    d["euro_equiv"] = np.minimum(d["euro_equiv"], d["option_price"].astype(float))
    d["euro_equiv"] = np.minimum(d["euro_equiv"], d["S_used"].astype(float))
    return d

def run_early_ex_premium(options: pd.DataFrame):
    df = options.copy()
    need = ["date","exdate","strike","stock_exdiv","YTM","risk_free","impl_volatility","option_price"]
    df = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    for c in ["date","exdate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
    df = add_div_yield(df)
    df = compute_euro_equivalent(df)
    df["american_premium"] = df["option_price"].astype(float)
    df["early_ex_premium"] = np.maximum(0.0, df["american_premium"] - df["euro_equiv"])
    df["moneyness"] = df["strike"] / df["S_used"]
    df = df[
        ((df["cp_flag"]=="C") & (df["moneyness"]>1.0)) |
        ((df["cp_flag"]=="P") & (df["moneyness"]<=1.0))
    ].copy()
    label_map = {
        pd.Timestamp("2020-01-17"): "Jan 17, 2020",
        pd.Timestamp("2020-03-20"): "Mar 20, 2020"
    }
    df["quote_date"] = df["date"].map(label_map)
    sns.set_context("talk"); sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    palette_cp = {"C": "tab:blue", "P": "tab:orange"}
    label_cp    = {"C": "Call (C)", "P": "Put (P)"}
    for ax, qd in zip(axes, ["Jan 17, 2020", "Mar 20, 2020"]):
        sub = df[df["quote_date"] == qd]
        sns.scatterplot(data=sub, x="moneyness", y="early_ex_premium", hue="cp_flag", palette=palette_cp, alpha=0.7, s=28, ax=ax)
        ax.set_title(qd)
        ax.set_xlabel("Moneyness (K / S_used)")
        ax.set_ylabel("Early-Exercise Premium (American − European)" if qd=="Jan 17, 2020" else "")
        ax.grid(True, linestyle="--", alpha=0.7)
        leg = ax.get_legend()
        if leg: leg.remove()
        handles, labels = ax.get_legend_handles_labels()
        labels = [label_cp.get(l, l) for l in labels]
        ax.legend(handles, labels, title="Option Type", loc="upper right", frameon=True, fontsize=9, title_fontsize=10)
    fig.suptitle("Early-Exercise Premium using BSM with σ = impl_volatility (American premium = option_price)")
    plt.tight_layout()
    plt.show()
    return df

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
        run_early_ex_premium(opts_df)
    else:
        raise SystemExit("provide options DataFrame path or import run_early_ex_premium and pass a DataFrame")
