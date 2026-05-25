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


def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = 'call') -> float:
    """First derivative of option price with respect to spot."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Second derivative of option price with respect to spot (same for call and put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensitivity to 1% change in volatility."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = 'call') -> float:
    """Daily time decay (price change per calendar day)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365.0


def rho(S: float, K: float, T: float, r: float, sigma: float,
        option_type: str = 'call') -> float:
    """Sensitivity to 1% change in risk-free rate."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def numerical_delta(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call', dS: float = 0.01) -> float:
    """Central difference approximation of delta."""
    return (bs_price(S + dS, K, T, r, sigma, option_type) -
            bs_price(S - dS, K, T, r, sigma, option_type)) / (2 * dS)


def numerical_gamma(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = 'call', dS: float = 0.01) -> float:
    """Central difference approximation of gamma."""
    return (bs_price(S + dS, K, T, r, sigma, option_type)
            - 2 * bs_price(S, K, T, r, sigma, option_type)
            + bs_price(S - dS, K, T, r, sigma, option_type)) / (dS ** 2)


def numerical_vega(S: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = 'call', d_sigma: float = 0.0001) -> float:
    """Central difference approximation of vega (per 1% vol change)."""
    return (bs_price(S, K, T, r, sigma + d_sigma, option_type) -
            bs_price(S, K, T, r, sigma - d_sigma, option_type)) / (2 * d_sigma * 100)
