"""Triple-barrier labels that mirror the exact trade the strategy places.

For signal bar i the trade is entered at the OPEN of bar i+1 with
TP = tp_atr * ATR_i and SL = sl_atr * ATR_i from the fill price, and a time exit after
`horizon` bars. Label = 1 if the TP is reached before the SL, else 0. If both are touched
inside one bar we assume the SL came first (conservative). Spread is included: longs fill
at ask and exit at bid, shorts fill at bid and exit at ask.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(
    df: pd.DataFrame,
    atr_values: pd.Series,
    spread: pd.Series,
    tp_atr: float,
    sl_atr: float,
    horizon: int,
) -> pd.DataFrame:
    o, h, l = (df[k].to_numpy() for k in ("open", "high", "low"))
    a = atr_values.to_numpy()
    sp = spread.to_numpy()
    n = len(df)
    y_long = np.full(n, np.nan)
    y_short = np.full(n, np.nan)

    for i in range(n - horizon - 1):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        tp_d, sl_d = tp_atr * a[i], sl_atr * a[i]
        e_long = o[i + 1] + sp[i + 1]
        e_short = o[i + 1]
        lt, ls = e_long + tp_d, e_long - sl_d
        st, ss = e_short - tp_d, e_short + sl_d
        long_res = short_res = None
        for j in range(i + 1, i + 1 + horizon):
            if long_res is None:
                if l[j] <= ls:
                    long_res = 0
                elif h[j] >= lt:
                    long_res = 1
            if short_res is None:
                ask_h, ask_l = h[j] + sp[j], l[j] + sp[j]
                if ask_h >= ss:
                    short_res = 0
                elif ask_l <= st:
                    short_res = 1
            if long_res is not None and short_res is not None:
                break
        y_long[i] = long_res or 0
        y_short[i] = short_res or 0

    return pd.DataFrame({"y_long": y_long, "y_short": y_short}, index=df.index)
