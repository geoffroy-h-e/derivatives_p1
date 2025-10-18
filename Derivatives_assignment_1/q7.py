# q7_borrow_fee.py
import sys
import numpy as np
import pandas as pd
from pathlib import Path

#  borrow fee approx
def compute_borrow_fee_approx(
    df: pd.DataFrame,
    target_days: int = 30,
    iv_col: str = "impl_volatility",  #contains the CRR implied vol
    r_col: str = "risk_free",
    spot_col: str = "stock_exdiv"
) -> pd.DataFrame:

    d = df.copy()

    #  Compute maturity
    for c in ["date", "exdate"]:
        d[c] = pd.to_datetime(d[c]).dt.tz_localize(None).dt.normalize()

    if "DTM" not in d.columns:
        d["DTM"] = (d["exdate"] - d["date"]).dt.days.astype(float)
    if "YTM" not in d.columns:
        d["YTM"] = d["DTM"] / 365.0

    if "cp_flag" not in d.columns:
        if "is_call" in d.columns:
            d["cp_flag"] = np.where(d["is_call"]==1, "C", "P")
        elif "option_type" in d.columns:
            d["cp_flag"] = d["option_type"].astype(str).str.upper().str[0]

    d = d.dropna(subset=["cp_flag", "strike", "YTM", iv_col])
    d["cp_flag"] = d["cp_flag"].astype(str).str.upper().str[0]
    d[iv_col] = d[iv_col].astype(float)

    out = []

    # Iterate by quote date
    for qdate in sorted(d["date"].unique()):
        dq = d[d["date"] == qdate]
        if dq.empty:
            continue

        # ATM options
        exp_choice = (
            dq[["exdate", "DTM"]]
            .drop_duplicates()
            .assign(diff=lambda x: (x["DTM"] - target_days).abs())
            .sort_values("diff")
        )
        if exp_choice.empty:
            continue
        ex_chosen = exp_choice.iloc[0]["exdate"]
        dqe = dq[dq["exdate"] == ex_chosen].copy()

        if dqe.empty:
            continue

        # T and r selection
        T = float(np.nanmedian(dqe["YTM"]))
        r = float(np.nanmedian(dqe[r_col])) if r_col in dqe.columns else 0.0
        T = max(T, 1e-12)

        # Spot and forward level selection
        if spot_col in dqe.columns:
            S_ref = float(np.nanmedian(dqe[spot_col]))
        elif "stock_price" in dqe.columns:
            S_ref = float(np.nanmedian(dqe["stock_price"]))
        else:
            S_ref = float(np.nanmedian(dqe["strike"]))
        K_star = S_ref * np.exp(r * T)

        #  Split calls & puts
        calls = dqe[dqe["cp_flag"] == "C"].copy()
        puts  = dqe[dqe["cp_flag"] == "P"].copy()
        if calls.empty or puts.empty:
            continue

        #  Interpolate σ at K*
        def iv_interp(df_side, is_call):
            sdf = df_side.dropna(subset=["strike", iv_col])
            sdf = sdf.sort_values("strike")

            # options filters for otm options
            sdf = sdf[sdf["strike"] >= K_star] if is_call else sdf[sdf["strike"] <= K_star]
            if sdf.empty:
                return np.nan

            Ks = sdf["strike"].values
            sig = sdf[iv_col].values

            if len(Ks) == 1:
                return float(sig[0])

            idx = np.searchsorted(Ks, K_star) #allows to find closest strikes
            i0 = max(0, idx - 1)
            i1 = min(len(Ks) - 1, idx)
            if i0 == i1:
                return float(sig[i0])

            K0, K1 = Ks[i0], Ks[i1]
            s0, s1 = sig[i0], sig[i1]
            w = (K_star - K0) / (K1 - K0)
            return float(s0 + w * (s1 - s0))

        sigma_c = iv_interp(calls, True)
        sigma_p = iv_interp(puts, False)

        if np.isnan(sigma_c) or np.isnan(sigma_p):
            continue

        #Approximation of H
        h_approx = -(sigma_c - sigma_p) / np.sqrt(2.0 * np.pi * T)

        out.append({
            "quote_date": pd.Timestamp(qdate).date(),
            "expiry": pd.Timestamp(ex_chosen).date(),
            "K_star": K_star,
            "sigma_c": sigma_c,
            "sigma_p": sigma_p,
            "T_years": T,
            "h_approx": h_approx
        })

    out_df = pd.DataFrame(out).sort_values(["quote_date", "expiry"]).reset_index(drop=True)
    return out_df

def _load_options_from_path(p):
    p = Path(p)
    if p.suffix.lower() in [".parquet",".pq",".parq"]:
        return pd.read_parquet(p)
    if p.suffix.lower() in [".csv",".txt"]:
        return pd.read_csv(p)
    if p.suffix.lower() in [".pkl",".pickle"]:
        return pd.read_pickle(p)
    raise ValueError("unsupported file type")

def run_q7_borrow_fee(options: pd.DataFrame, target_days=30, iv_col="impl_volatility", r_col="risk_free", spot_col="stock_exdiv"):
    bf = compute_borrow_fee_approx(options, target_days=target_days, iv_col=iv_col, r_col=r_col, spot_col=spot_col)
    print(bf)
    return bf

if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        opts = _load_options_from_path(args[0])
        run_q7_borrow_fee(opts)
    else:
        raise SystemExit("provide options DataFrame path or import run_q7_borrow_fee and pass a DataFrame")
