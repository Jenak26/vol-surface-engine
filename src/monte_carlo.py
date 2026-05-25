import numpy as np
from src.black_scholes import bs_price


def simulate_gbm_paths(S: float, T: float, r: float, sigma: float,
                       n_sims: int = 100_000, n_steps: int = 252,
                       seed: int = 42) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion price paths.

    Returns array of shape (n_sims, n_steps + 1), where column 0 is the
    initial spot price and column n_steps is the terminal price.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_sims, n_steps))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def mc_european(S: float, K: float, T: float, r: float, sigma: float,
                option_type: str = 'call', n_sims: int = 100_000,
                seed: int = 42) -> float:
    """Baseline European option price via Monte Carlo (no variance reduction)."""
    paths = simulate_gbm_paths(S, T, r, sigma, n_sims=n_sims, n_steps=1, seed=seed)
    ST = paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    return float(np.exp(-r * T) * np.mean(payoffs))


def mc_antithetic(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = 'call', n_sims: int = 100_000,
                  seed: int = 42) -> float:
    """
    European MC with antithetic variates.
    Uses n_sims/2 pairs (Z, -Z) so total path count equals n_sims.
    """
    half = n_sims // 2
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(half)
    log_drift = (r - 0.5 * sigma ** 2) * T
    ST_pos = S * np.exp(log_drift + sigma * np.sqrt(T) * Z)
    ST_neg = S * np.exp(log_drift - sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = 0.5 * (np.maximum(ST_pos - K, 0.0) + np.maximum(ST_neg - K, 0.0))
    else:
        payoffs = 0.5 * (np.maximum(K - ST_pos, 0.0) + np.maximum(K - ST_neg, 0.0))
    return float(np.exp(-r * T) * np.mean(payoffs))


def mc_control_variate(S: float, K: float, T: float, r: float, sigma: float,
                       option_type: str = 'call', n_sims: int = 100_000,
                       seed: int = 42) -> float:
    """
    European MC with control variate.
    Control: discounted terminal stock price (mean = S analytically).
    beta is estimated from the sample covariance.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)
    discounted_ST = np.exp(-r * T) * ST
    # Optimal beta minimises variance of (payoff - beta * control)
    control = discounted_ST - S  # zero-mean: E[discounted_ST] = S
    cov_matrix = np.cov(np.exp(-r * T) * payoffs, control)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    adjusted = np.exp(-r * T) * payoffs - beta * control
    return float(np.mean(adjusted))
