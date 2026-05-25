import numpy as np


def crr_american_put(S: float, K: float, T: float, r: float, sigma: float,
                     n_steps: int = 200) -> float:
    """
    American put price via Cox-Ross-Rubinstein (CRR) binomial tree.

    The CRR parameterisation ensures the tree recombines:
        u = exp(sigma * sqrt(dt))
        d = 1 / u
        p = (exp(r*dt) - d) / (u - d)

    At each node the holder chooses the greater of immediate exercise
    (K - S) and discounted continuation.

    Args:
        S: spot price
        K: strike price
        T: time to expiry in years
        r: continuously compounded risk-free rate
        sigma: annualised volatility
        n_steps: number of time steps (higher = more accurate, O(n^2) cost)

    Returns:
        American put price
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices: S * u^j * d^(n-j) for j = 0..n
    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))
    values = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        j = np.arange(i + 1)
        S_node = S * (u ** j) * (d ** (i - j))
        continuation = discount * (p * values[1:i + 2] + (1 - p) * values[:i + 1])
        intrinsic = np.maximum(K - S_node, 0.0)
        values = np.maximum(intrinsic, continuation)

    return float(values[0])


def crr_european_put(S: float, K: float, T: float, r: float, sigma: float,
                     n_steps: int = 200) -> float:
    """
    European put price via CRR binomial tree (no early exercise).
    Converges to Black-Scholes as n_steps → ∞.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))
    values = np.maximum(K - ST, 0.0)

    for _ in range(n_steps):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

    return float(values[0])
