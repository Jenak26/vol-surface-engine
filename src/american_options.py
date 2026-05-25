import numpy as np
from src.monte_carlo import simulate_gbm_paths


def _laguerre_basis(x: np.ndarray) -> np.ndarray:
    """
    Three Laguerre polynomial basis functions scaled to strike level.
    These are standard basis functions used in the original LSM paper
    (Longstaff & Schwartz 2001).
    """
    u = x  # raw stock prices as regressor
    col1 = np.exp(-u / 2)
    col2 = np.exp(-u / 2) * (1 - u)
    col3 = np.exp(-u / 2) * (1 - 2 * u + 0.5 * u ** 2)
    return np.column_stack([col1, col2, col3])


def lsm_american_put(S: float, K: float, T: float, r: float, sigma: float,
                     n_sims: int = 50_000, n_steps: int = 50,
                     seed: int = 42) -> float:
    """
    American put price via Longstaff-Schwartz Monte Carlo (LSM).

    Uses antithetic variates for variance reduction (half the paths are
    antithetic mirrors, so effective path count = n_sims).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Antithetic path generation
    half = n_sims // 2
    Z = rng.standard_normal((half, n_steps))
    Z_full = np.concatenate([Z, -Z], axis=0)  # shape: (n_sims, n_steps)

    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = S
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_full[:, t]
        )

    # Initialise cashflows at expiry
    cashflows = np.maximum(K - paths[:, -1], 0.0)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - paths[:, t], 0.0)
        itm = intrinsic > 0

        if itm.sum() < 4:  # need at least 4 points to regress
            cashflows *= discount
            continue

        X = paths[itm, t] / K  # normalise to avoid numerical issues
        Y = cashflows[itm] * discount  # discounted future cashflow

        basis = _laguerre_basis(X)
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation = basis @ coeffs

        exercise = intrinsic[itm] > continuation
        cashflows[itm] = np.where(exercise, intrinsic[itm], cashflows[itm] * discount)
        cashflows[~itm] *= discount

    return float(discount * np.mean(cashflows))


def compute_exercise_boundary(K: float, T: float, r: float, sigma: float,
                               n_steps: int = 50, n_sims: int = 20_000,
                               seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the early exercise boundary for an American put.

    At each timestep, find the critical stock price S* such that
    immediate exercise (K - S*) equals the continuation value.

    Returns:
        times: array of time points (length n_steps)
        boundary: critical stock prices S*(t) at each time
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    half = n_sims // 2
    Z = rng.standard_normal((half, n_steps))
    Z_full = np.concatenate([Z, -Z], axis=0)

    paths = np.empty((n_sims, n_steps + 1))
    paths[:, 0] = K  # start at-the-money
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_full[:, t]
        )

    cashflows = np.maximum(K - paths[:, -1], 0.0)
    boundary = np.empty(n_steps)

    for t in range(n_steps - 1, -1, -1):
        intrinsic = np.maximum(K - paths[:, t], 0.0)
        itm = intrinsic > 0

        if itm.sum() >= 4:
            X = paths[itm, t] / K
            Y = cashflows[itm] * discount
            basis = _laguerre_basis(X)
            coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
            continuation = basis @ coeffs
            exercise = intrinsic[itm] > continuation
            # Critical price: highest S where exercise is optimal
            exercise_prices = paths[itm, t][exercise]
            boundary[t] = np.max(exercise_prices) if exercise.any() else 0.0
            cashflows[itm] = np.where(exercise, intrinsic[itm], cashflows[itm] * discount)
        else:
            boundary[t] = 0.0

        cashflows[~itm] *= discount

    times = np.linspace(0, T, n_steps)
    return times, boundary
