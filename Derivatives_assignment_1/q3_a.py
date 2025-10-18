
# q3_smiles.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import fsolve

#  columns
DATE = 'date'
EXDATE = 'exdate'
K = 'strike'
S = 'stock_price'
S_EX = 'stock_exdiv'    
T = 'YTM'                
RF = 'risk_free'       
DTM = 'DTM'             
OPTION_PRICE = 'option_price'
F_IMP  = 'implied_forward_price'

# Implied Volatility 
IV_QUOTED = 'impl_volatility'      
IV_BMS_QUOTED = 'implied_vol_bms' 

# Option specific columns
CP = 'is_call'       

# IV calculation
def BMS_price(S0, K_val, T_val, r, y, sigma, is_call):
    """Black–Merton–Scholes price with continuous dividend yield y."""
    if T_val <= 0:
        discS = S0*np.exp(-y*T_val)
        discK = K_val*np.exp(-r*T_val)
        return max(0.0, discS - discK) if is_call else max(0.0, discK - discS)

    sigma = max(1e-10, float(sigma))
    d1 = (np.log(S0/K_val) + (r - y + 0.5*sigma**2)*T_val) / (sigma*np.sqrt(T_val))
    d2 = d1 - sigma*np.sqrt(T_val)
    if is_call:
        return np.exp(-y*T_val)*S0*norm.cdf(d1) - K_val*np.exp(-r*T_val)*norm.cdf(d2)
    else:
        return K_val*np.exp(-r*T_val)*norm.cdf(-d2) - np.exp(-y*T_val)*S0*norm.cdf(-d1)

def vega_BMS(S0, K_val, T_val, r, y, sigma):
    """BMS vega under continuous dividend yield y."""
    if T_val <= 0:
        return 0.0
    sigma = max(1e-10, float(sigma))
    d1 = (np.log(S0/K_val) + (r - y + 0.5*sigma**2)*T_val) / (sigma*np.sqrt(T_val))
    return np.exp(-y*T_val)*S0*np.sqrt(T_val)*norm.pdf(d1)

def IV_newton(mkt_p, S0, K_val, T_val, r, y, is_call, x0=0.2, tol=1e-8, maxit=100):
    """Invert BMS using Newton–Raphson with an fsolve fallback."""
    sigma = max(1e-4, float(x0))
    for _ in range(maxit):
        px = BMS_price(S0, K_val, T_val, r, y, sigma, is_call)
        diff = px - mkt_p
        if abs(diff) < tol:
            return float(max(1e-8, sigma))
        v = vega_BMS(S0, K_val, T_val, r, y, sigma)
        if v <= 1e-12:
            break
        sigma = max(1e-8, sigma - diff / v)

    root = fsolve(lambda s: BMS_price(S0, K_val, T_val, r, y, s, is_call) - mkt_p, sigma, xtol=tol)
    return float(max(1e-8, root[0]))


#  Data preparation 
def infer_y_row(row):
    """y = r - ln(F/S)/T   (cash-and-carry using implied_forward_price)."""
    T_val = float(row[T])
    F_val = float(row[F_IMP]) if pd.notna(row[F_IMP]) else np.nan
    if (T_val <= 0) or (not np.isfinite(F_val)) or (F_val <= 0):
        return 0.0
    return float(row[RF] - np.log(F_val/row[S]) / T_val)

def build_otm_options(options: pd.DataFrame) -> pd.DataFrame:
    """
    From the full options DF, build the OTM subset and compute BMS+y IV
    (column 'cmptd_IV'). Returns the otm_options DF
    """
    req = [DATE, EXDATE, CP, K, S, S_EX, T, RF, F_IMP, OPTION_PRICE, IV_BMS_QUOTED, IV_QUOTED]
    df = options[req].dropna().copy()
    df[DATE]   = pd.to_datetime(df[DATE])
    df[EXDATE] = pd.to_datetime(df[EXDATE])

    # dividend yield 
    df['y'] = df.apply(infer_y_row, axis=1)

    # moneyness
    df['M'] = df[K] / df[S_EX]

    # OTM mask
    otm = df[((df[CP] == 1) & (df['M'] > 1.0)) | ((df[CP] == 0) & (df['M'] <= 1.0))].copy()

    # compute IV
    def _iv(r):
        is_call = bool(int(r[CP]) == 1)
        x0 = float(r[IV_BMS_QUOTED]) if pd.notna(r[IV_BMS_QUOTED]) else 0.2
        return IV_newton(
            mkt_p=float(r[OPTION_PRICE]),
            S0=float(r[S]),
            K_val=float(r[K]),
            T_val=float(r[T]),
            r=float(r[RF]),
            y=float(r['y']),
            is_call=is_call,
            x0=x0
        )
    otm['cmptd_IV'] = otm.apply(_iv, axis=1)

    # Structuring
    otm_options = otm[[DATE, EXDATE, CP, K, S_EX, 'M', OPTION_PRICE, RF, 'y', T, 'cmptd_IV', IV_BMS_QUOTED, IV_QUOTED]].copy()
    return otm_options


#  Plotting 
def plot_q3_smiles(otm_options: pd.DataFrame):
    """
    1×2 figure (left/right panels per assignment):
      Left  panel: (Jan 17 → Feb 14) and (Mar 20 → Apr 17) in 2020
      Right panel: (Jan 17 → Jul 17) and (Mar 20 → Oct 16) in 2020

    Line = provider's IV (IV_BMS_QUOTED), dots =  computed IV (cmptd_IV).
    """
    pairs_left  = [
        (pd.Timestamp(2020, 1, 17), pd.Timestamp(2020, 2, 14)),
        (pd.Timestamp(2020, 3, 20), pd.Timestamp(2020, 4, 17)),
    ]
    pairs_right = [
        (pd.Timestamp(2020, 1, 17), pd.Timestamp(2020, 7, 17)),
        (pd.Timestamp(2020, 3, 20), pd.Timestamp(2020, 10, 16)),
    ]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    expiry_cmap = plt.colormaps.get_cmap('tab10')

    def _plot_panel(ax, pairs, title):
        for i, (qdate, edate) in enumerate(pairs):
            sub = otm_options[(otm_options[DATE] == qdate) & (otm_options[EXDATE] == edate)].copy()
            lbl = f"{qdate.date()} → {edate.date()}"
            color = expiry_cmap(i)

            if sub.empty:
                ax.text(0.5, 0.5, f"No data for\n{lbl}", ha='center', va='center', transform=ax.transAxes)
                continue

            sub = sub.sort_values('M')
            # Line: provider's quoted IV
            ax.plot(sub['M'], sub[IV_BMS_QUOTED], lw=2, color=color, label=f"{lbl} — Quoted IV")
            # Dots: computed IV
            ax.scatter(sub['M'], sub['cmptd_IV'], s=45, color=color, edgecolor='k', linewidths=0.5,
                       alpha=0.95, zorder=3, label=f"{lbl} — Computed IV")

        ax.set_title(title, pad=10)
        ax.set_xlabel("Moneyness")
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=10, loc='upper right', frameon=True)

    _plot_panel(axes[0], pairs_left,  "Near-term smiles")
    _plot_panel(axes[1], pairs_right, "Longer-term smiles")

    axes[0].set_ylabel("Implied Volatility (BMS)")
    fig.suptitle("Q3 — Implied Volatility Smiles (OTM only)\n"
                 "Left: Jan17→Feb14 & Mar20→Apr17 • Right: Jan17→Jul17 & Mar20→Oct16",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


__all__ = [
    'DATE','EXDATE','K','S','S_EX','T','RF','DTM','OPTION_PRICE','F_IMP',
    'IV_QUOTED','IV_BMS_QUOTED','CP',
    'BMS_price','vega_BMS','IV_newton',
    'infer_y_row','build_otm_options','plot_q3_smiles'
]
