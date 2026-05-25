import numpy as np
from src.black_scholes import bs_price, vega as bs_vega


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = 'call', tol: float = 1e-6,
                max_iter: int = 200) -> float | None:
    """
    Implied volatility via Newton-Raphson iteration.

    Returns the sigma such that BS_price(sigma) == market_price.
    Returns None if the solver fails to converge (e.g., deep OTM options
    where vega → 0 or market price is below intrinsic value).
    """
    # Intrinsic value check — no real IV exists below intrinsic
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)
    if market_price < intrinsic - 1e-4:
        return None

    sigma = 0.20  # initial guess

    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma) * 100  # vega() returns per-1%, undo the /100

        if abs(v) < 1e-10:
            return None  # vega too small, Newton-Raphson unstable

        sigma -= (price - market_price) / v

        if sigma <= 1e-6:
            sigma = 1e-6  # clamp to prevent negative vol

        if abs(bs_price(S, K, T, r, sigma, option_type) - market_price) < tol:
            return float(sigma)

    return None  # did not converge within max_iter
