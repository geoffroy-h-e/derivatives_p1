# q4_a.py 
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Helpers 
def _norm_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["date", "exdate"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize(None).dt.normalize()
    return df

def _standardize_cp_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a string cp_flag ('C'/'P') column exists."""
    df = df.copy()
    if "cp_flag" in df.columns:
        # Normalize values
        df["cp_flag"] = df["cp_flag"].map(lambda x: "C" if str(x).upper().startswith("C") or str(x)=="1" else ("P" if str(x).upper().startswith("P") or str(x)=="0" else x))
        return df
    # Derive from is_call
    for cand in ["is_call", "call_put", "cp"]:
        if cand in df.columns:
            def to_cp(v):
                if isinstance(v, str):
                    v = v.strip().upper()
                    if v in ("C","CALL","CALLS"): return "C"
                    if v in ("P","PUT","PUTS"): return "P"
                try:
                    return "C" if int(v)==1 else "P"
                except Exception:
                    return v
            df["cp_flag"] = df[cand].map(to_cp)
            return df
    raise KeyError("Neither 'cp_flag' nor a usable call/put indicator (e.g., 'is_call') found in DataFrame.")

def _coalesce_columns(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    Create/rename columns so df has each target name, using the first available alias.
    targets: {canonical: [aliases...]}
    """
    df = df.copy()
    for canon, aliases in targets.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                df[canon] = df[a]
                break
        if canon not in df.columns:
            pass
    return df

# Build custom fonctions for IV
def build_iv_diff_df(options: pd.DataFrame, otm_options: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataframe matching the user's code, but tolerate older schemas:
      - cp_flag or is_call => cp_flag ('C'/'P')
      - iv_bsm or implied_vol_bms => iv_bsm
      - stock_exdiv or S_EX (fallback to stock_price if necessary)
    """
    opts = _norm_dates(options)
    otm  = _norm_dates(otm_options)

    # Standardize call/put flag
    opts = _standardize_cp_flag(opts)
    otm  = _standardize_cp_flag(otm)

    # column names
    opts = _coalesce_columns(opts, {
        "impl_volatility": ["implied_vol_bms", "impl_vol", "iv_provider"]
    })
    otm = _coalesce_columns(otm, {
        "iv_bsm": ["implied_vol_bms", "iv_bms", "implied_vol_bsm"]
    })
    # Spot/ex-div
    opts = _coalesce_columns(opts, {
        "stock_exdiv": ["S_EX", "stock_ex_div", "S-hat", "S_hat", "S_EXDIV", "stock_price"]
    })
    otm = _coalesce_columns(otm, {
        "stock_exdiv": ["S_EX", "stock_ex_div", "S-hat", "S_hat", "S_EXDIV", "stock_price"]
    })

    missing_opts = [c for c in ["date","exdate","cp_flag","strike","impl_volatility","stock_exdiv"] if c not in opts.columns]
    missing_otm  = [c for c in ["date","exdate","cp_flag","strike","iv_bsm","stock_exdiv"] if c not in otm.columns]
    if missing_opts:
        raise KeyError(f"`options` is missing required columns after standardization: {missing_opts}")
    if missing_otm:
        raise KeyError(f"`otm_options` is missing required columns after standardization: {missing_otm}")

    keys = ["date", "exdate", "cp_flag", "strike"]
    df = (
        otm[keys + ["stock_exdiv", "iv_bsm"]]
        .merge(opts[keys + ["impl_volatility"]], on=keys, how="left")
        .dropna(subset=["iv_bsm", "impl_volatility", "stock_exdiv"])
    )

    # Same moneyness definition
    df["moneyness"] = df["strike"] / df["stock_exdiv"]

    # OTM filter
    df = df[((df["cp_flag"] == "P") & (df["moneyness"] <= 1.0)) |
            ((df["cp_flag"] == "C") & (df["moneyness"] > 1.0))]

    # IV difference (%)
    df["iv_diff_pct"] = 100 * (df["impl_volatility"] / df["iv_bsm"] - 1)
    return df

#  Plotting 
def plot_iv_diff(df: pd.DataFrame):

    df_jan17 = df[df["date"] == pd.Timestamp("2020-01-17")].copy()
    df_mar20 = df[df["date"] == pd.Timestamp("2020-03-20")].copy()

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    palette_cp = {"C": "steelblue", "P": "orange"}
    label_map = {"P": "Put (P)", "C": "Call (C)"}

    #  Jan 17, 2020 
    sns.scatterplot(
        data=df_jan17,
        x="moneyness",
        y="iv_diff_pct",
        hue="cp_flag",
        palette=palette_cp,
        alpha=0.6,
        s=25,
        ax=axes[0]
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Jan 17, 2020")
    axes[0].set_xlabel("Moneyness")
    axes[0].set_ylabel("IV Difference (%)")

    leg = axes[0].get_legend()
    if leg: leg.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [label_map.get(l, l) for l in labels]
    axes[0].legend(
        handles, labels,
        title="Option Type",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    #  Mar 20, 2020
    sns.scatterplot(
        data=df_mar20,
        x="moneyness",
        y="iv_diff_pct",
        hue="cp_flag",
        palette=palette_cp,
        alpha=0.6,
        s=25,
        ax=axes[1]
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Mar 20, 2020")
    axes[1].set_xlabel("Moneyness")
    axes[1].set_ylabel("")

    leg = axes[1].get_legend()
    if leg: leg.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    labels = [label_map.get(l, l) for l in labels]
    axes[1].legend(
        handles, labels,
        title="Option Type",
        loc="upper right",
        frameon=True,
        fontsize=9,
        title_fontsize=10
    )

    fig.suptitle("Provider IV vs BSM IV (OTM Options)")
    plt.tight_layout()
    return fig, axes

# convenience 
def plot_from_raw(options: pd.DataFrame, otm_options: pd.DataFrame):
    df = build_iv_diff_df(options, otm_options)
    return plot_iv_diff(df)

# 
if __name__ == "__main__":
    if "options" not in globals() or "otm_options" not in globals():
        raise RuntimeError("Please define `options` and `otm_options` in the current session before running q4_a.py directly.")
    df__ = build_iv_diff_df(options, otm_options)
    plot_iv_diff(df__)
