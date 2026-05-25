# src/vol_surface.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d


def _svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                         m: float, sigma: float) -> np.ndarray:
    """
    SVI (Stochastic Volatility Inspired) parametrization of total variance w = sigma²*T.
    Gatheral (2004): w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _fit_svi_slice(log_moneyness: np.ndarray, total_var: np.ndarray) -> np.ndarray:
    """
    Fit SVI parameters to a single expiry slice via least-squares.
    Returns [a, b, rho, m, sigma].
    """
    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        w_fit = _svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        if np.any(w_fit <= 0):
            return 1e10
        return float(np.sum((w_fit - total_var) ** 2))

    near_atm = total_var[np.abs(log_moneyness) < 0.05]
    if len(near_atm) == 0:
        near_atm = total_var  # fallback if no near-ATM strikes
    atm_var = float(np.mean(near_atm))
    x0 = [atm_var * 0.8, 0.1, -0.3, 0.0, 0.2]
    bounds = [(1e-6, None), (1e-6, None), (-0.999, 0.999), (-1.0, 1.0), (1e-4, None)]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12})
    return result.x


class VolatilitySurface:
    """
    Implied volatility surface fitted from market option prices.

    Fits SVI parametrization per expiry slice, then linearly interpolates
    total variance between slices for arbitrary (T, k) queries.
    """

    def __init__(self, df: pd.DataFrame, spot: float, r: float):
        self.spot = spot
        self.r = r
        self._fit(df)

    def _fit(self, df: pd.DataFrame):
        self._slices: dict[float, np.ndarray] = {}
        for T_val, group in df.groupby('T'):
            k = group['log_moneyness'].values
            iv = group['computed_iv'].values
            total_var = iv ** 2 * T_val
            if len(k) < 5:
                continue
            params = _fit_svi_slice(k, total_var)
            self._slices[float(T_val)] = params
        self._T_vals = np.array(sorted(self._slices.keys()))

    def get_iv(self, log_moneyness: float, T: float) -> float:
        """Interpolated implied vol at arbitrary (log_moneyness, T)."""
        k = np.array([log_moneyness])
        T = float(T)

        if T <= self._T_vals[0]:
            params = self._slices[self._T_vals[0]]
            w = _svi_total_variance(k, *params)[0]
            return float(np.sqrt(max(w, 1e-8) / self._T_vals[0]))

        if T >= self._T_vals[-1]:
            params = self._slices[self._T_vals[-1]]
            w = _svi_total_variance(k, *params)[0]
            return float(np.sqrt(max(w, 1e-8) / self._T_vals[-1]))

        # Linear interpolation in total variance between bracketing slices
        idx = np.searchsorted(self._T_vals, T) - 1
        T_lo, T_hi = self._T_vals[idx], self._T_vals[idx + 1]
        alpha = (T - T_lo) / (T_hi - T_lo)

        w_lo = _svi_total_variance(k, *self._slices[T_lo])[0]
        w_hi = _svi_total_variance(k, *self._slices[T_hi])[0]
        w = (1 - alpha) * w_lo + alpha * w_hi
        return float(np.sqrt(max(w, 1e-8) / T))

    def get_iv_grid(self, moneyness_grid: np.ndarray,
                    T_grid: np.ndarray) -> np.ndarray:
        """Returns IV surface on a (len(T_grid), len(moneyness_grid)) grid."""
        return np.array([
            [self.get_iv(k, T) for k in moneyness_grid]
            for T in T_grid
        ])
