# q4.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq

# BSM price formula for call/put
def bs_price(S, K, r, tau, sigma, cp):
    if sigma <= 0 or tau <= 0:
        intrinsic = max(0.0, (S - K) if cp == "C" else (K - S))
        return intrinsic
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Solve implied volatility
def implied_vol(price, S, K, r, tau, cp, vol_lo=1e-6, vol_hi=5.0):
    forward = S * np.exp(r * tau)
    intrinsic = max(0.0, (S - K) if cp == "C" else (K - S))
    upper_bound = S if cp == "C" else K * np.exp(-r * tau)
    if not (intrinsic <= price <= upper_bound + 1e-8):
        return np.nan
    def f(sig):
        return bs_price(S, K, r, tau, sig, cp) - price
    try:
        lo, hi = vol_lo, vol_hi
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            for hi in (7.5, 10.0):
                if f(lo) * f(hi) <= 0:
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, maxiter=200, xtol=1e-8)
    except Exception:
        return np.nan

def _norm_dates(df):
    df = df.copy()
    for c in ["date", "exdate"]:
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
    return df


# Main function for IV comparison
def run_q4_iv_comparison(options):
    options = options.copy()
    options["M"] = options["strike"] / options["stock_exdiv"]

    mask_otm = ((options["cp_flag"] == "P") & (options["M"] <= 1.0)) | \
               ((options["cp_flag"] == "C") & (options["M"] > 1.0))
    otm_options = options.loc[mask_otm].copy()

    def _row_iv(row):
        return implied_vol(
            price=row["option_price"],
            S=row["stock_exdiv"],
            K=row["strike"],
            r=row["risk_free"],
            tau=row["YTM"],
            cp=row["cp_flag"]
        )
    otm_options["iv_bsm"] = otm_options.apply(_row_iv, axis=1)

    opts = _norm_dates(options)
    otm  = _norm_dates(otm_options)

    keys = ["date", "exdate", "cp_flag", "strike"]
    df = (
        otm[keys + ["stock_exdiv", "iv_bsm"]]
        .merge(opts[keys + ["impl_volatility"]], on=keys, how="left")
        .dropna(subset=["iv_bsm", "impl_volatility", "stock_exdiv"])
    )

    df["moneyness"] = df["strike"] / df["stock_exdiv"]
    df = df[((df["cp_flag"] == "P") & (df["moneyness"] <= 1.0)) |
            ((df["cp_flag"] == "C") & (df["moneyness"] > 1.0))]
    df["iv_diff_pct"] = 100 * (df["impl_volatility"] / df["iv_bsm"] - 1)

    df_jan17 = df[df['date'] == pd.Timestamp("2020-01-17")].copy()
    df_mar20 = df[df['date'] == pd.Timestamp("2020-03-20")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    palette = {"P": "orange", "C": "steelblue"}

    sns.scatterplot(data=df_jan17, x="moneyness", y="iv_diff_pct",
                    hue="cp_flag", palette=palette, alpha=0.6, s=25, ax=axes[0])
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Jan 17, 2020")
    axes[0].set_xlabel("Moneyness")
    axes[0].set_ylabel("IV Difference (%)")
    axes[0].legend(title="Type", labels=["Put", "Call"])

    sns.scatterplot(data=df_mar20, x="moneyness", y="iv_diff_pct",
                    hue="cp_flag", palette=palette, alpha=0.6, s=25, ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Mar 20, 2020")
    axes[1].set_xlabel("Moneyness")
    axes[1].set_ylabel("")
    axes[1].legend(title="Type", labels=["Put", "Call"])

    fig.suptitle("Provider IV vs BSM IV (OTM options)")
    plt.tight_layout()
    plt.show()
    return df
