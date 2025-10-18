# q3_b.py

import numpy as np
import matplotlib.pyplot as plt
import q3_a as q3a  


def replot_log_forward(otm_options: "pd.DataFrame") -> None:
    """
    Replots the Q3 implied-volatility smiles using log-forward moneyness:
        m = ln(K / F),  where  F = S_ex * exp((r - y) * T)
    """
    #  Comute log-forward moneyness
    df = otm_options.copy()
    F = df[q3a.S_EX] * np.exp((df[q3a.RF] - df["y"]) * df[q3a.T])
    df["M"] = np.log(df[q3a.K] / F)

    _old_show = plt.show
    plt.show = lambda *a, **k: None

    q3a.plot_q3_smiles(df)

    fig = plt.gcf()
    for ax in fig.axes:
        ax.set_xlabel("log-forward moneyness  m = ln(K/F)")
    fig.suptitle(
        "Q3 — Implied Volatility Smiles (OTM only) — log-forward moneyness",
        fontsize=15, y=1.02
    )
    plt.tight_layout()

    plt.show = _old_show
    plt.show()


#  test entry point
if __name__ == "__main__":
    import pandas as pd
    options = pd.read_csv("your_options_file.csv", parse_dates=["date", "exdate"])
    otm_options = q3a.build_otm_options(options)
    replot_log_forward(otm_options)
