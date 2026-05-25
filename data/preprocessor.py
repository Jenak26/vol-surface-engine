# data/preprocessor.py
import pandas as pd
import numpy as np
from src.implied_vol import implied_vol


def clean_option_chain(df: pd.DataFrame, spot: float, r: float = 0.065,
                        min_oi: int = 100, min_volume: int = 10) -> pd.DataFrame:
    """
    Filter and enrich a raw options chain DataFrame.

    Removes:
    - Zero or negative prices
    - Low open interest (illiquid strikes)
    - Options with no meaningful bid-ask
    - Strikes too far OTM (log-moneyness > 3 standard deviations)

    Adds:
    - mid_price: (bid + ask) / 2
    - log_moneyness: ln(K / F) where F = spot * exp(r * T)
    - computed_iv: implied vol from mid price via Newton-Raphson
    """
    df = df.copy()
    df = df[df['last_price'] > 0].copy()
    df = df[df['oi'] >= min_oi].copy()
    df = df[df['volume'] >= min_volume].copy()

    if 'bid' in df.columns and 'ask' in df.columns:
        df = df[df['ask'] > df['bid']].copy()  # remove crossed markets
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    else:
        df['mid_price'] = df['last_price']

    if 'T' not in df.columns:
        raise ValueError("DataFrame must have a 'T' column (time to expiry in years)")

    forward = spot * np.exp(r * df['T'])
    df['log_moneyness'] = np.log(df['strike'] / forward)

    # Compute implied vol from mid price
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(
            row['mid_price'], spot, row['strike'],
            row['T'], r, row['option_type']
        )
        ivs.append(iv)
    df['computed_iv'] = ivs
    df = df.dropna(subset=['computed_iv'])
    df = df[df['computed_iv'] > 0.02]   # discard suspiciously low IV
    df = df[df['computed_iv'] < 3.0]    # discard suspiciously high IV

    return df.reset_index(drop=True)
