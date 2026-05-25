import numpy as np


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
