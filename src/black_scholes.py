import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.

    Args:
        S: current spot price
        K: strike price
        T: time to expiry in years
        r: continuously compounded risk-free rate
        sigma: annualised volatility

    Returns:
        tuple of (d1, d2)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = 'call') -> float:
    """
    Black-Scholes European option price.

    Args:
        S: current spot price
        K: strike price
        T: time to expiry in years
        r: continuously compounded risk-free rate
        sigma: annualised volatility
        option_type: 'call' or 'put'

    Returns:
        option price
    """
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
