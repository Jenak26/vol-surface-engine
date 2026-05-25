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


def _svi_dw_dk(k: np.ndarray, a: float, b: float, rho: float,
               m: float, sigma: float) -> np.ndarray:
    """∂w/∂k for SVI total variance (analytical)."""
    z = k - m
    sqrt_term = np.sqrt(z ** 2 + sigma ** 2)
    return b * (rho + z / sqrt_term)


def _svi_d2w_dk2(k: np.ndarray, a: float, b: float, rho: float,
                 m: float, sigma: float) -> np.ndarray:
    """∂²w/∂k² for SVI total variance (analytical)."""
    z = k - m
    sqrt_term = np.sqrt(z ** 2 + sigma ** 2)
    return b * sigma ** 2 / sqrt_term ** 3


def _fit_svi_slice(log_moneyness: np.ndarray, total_var: np.ndarray) -> np.ndarray:
    """
    Fit SVI parameters to a single expiry slice via least-squares,
    penalising parameter sets that violate butterfly arbitrage.
    Returns [a, b, rho, m, sigma].
    """
    def objective(params):
        a, b, rho, m, sigma = params
        if b < 0 or sigma <= 0 or abs(rho) >= 1:
            return 1e10
        w_fit = _svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        if np.any(w_fit <= 0):
            return 1e10
        
        # Penalise butterfly arbitrage (g < 0) on a dense log-moneyness grid
        k_check = np.linspace(-0.4, 0.4, 30)
        w = _svi_total_variance(k_check, a, b, rho, m, sigma)
        wk = _svi_dw_dk(k_check, a, b, rho, m, sigma)
        wkk = _svi_d2w_dk2(k_check, a, b, rho, m, sigma)
        
        w_safe = np.maximum(w, 1e-12)
        A = 1.0 - k_check * wk / (2.0 * w_safe)
        g = A ** 2 - (wk ** 2 / 4.0) * (1.0 / w_safe + 0.25) + wkk / 2.0
        
        violations = np.sum(np.maximum(-g, 0.0))
        penalty = 1e4 * violations
        
        return float(np.sum((w_fit - total_var) ** 2)) + penalty

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

    Fits SVI parametrization per expiry slice, then uses a natural cubic spline
    in the time dimension to interpolate total variance, ensuring C2 smoothness
    and exact analytical time derivatives.
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

    def get_total_variance_and_derivative(self, log_moneyness: float, T: float) -> tuple[float, float]:
        """
        Returns (total_variance, dw_dT) at (log_moneyness, T) using a natural cubic spline
        in the time dimension to ensure C2 smoothness and exact analytical time derivatives.
        """
        k = np.array([log_moneyness])
        T = float(T)

        # If there are fewer than 3 slices, fall back to linear interpolation
        if len(self._T_vals) < 3:
            if T <= self._T_vals[0]:
                w = _svi_total_variance(k, *self._slices[self._T_vals[0]])[0]
                return float(w), 0.0
            if T >= self._T_vals[-1]:
                w = _svi_total_variance(k, *self._slices[self._T_vals[-1]])[0]
                return float(w), 0.0

            idx = np.searchsorted(self._T_vals, T) - 1
            T_lo, T_hi = self._T_vals[idx], self._T_vals[idx + 1]
            w_lo = _svi_total_variance(k, *self._slices[T_lo])[0]
            w_hi = _svi_total_variance(k, *self._slices[T_hi])[0]
            dw_dT = (w_hi - w_lo) / (T_hi - T_lo)
            w = w_lo + dw_dT * (T - T_lo)
            return float(w), float(dw_dT)

        # Cubic spline interpolation in time dimension
        from scipy.interpolate import CubicSpline
        w_slices = []
        for T_i in self._T_vals:
            w_i = _svi_total_variance(k, *self._slices[T_i])[0]
            w_slices.append(w_i)

        # Create natural cubic spline
        spline = CubicSpline(self._T_vals, w_slices, bc_type='natural')

        w = float(spline(T))
        dw_dT = float(spline(T, nu=1))

        # Ensure total variance is strictly non-negative
        w = max(w, 1e-8)

        return w, dw_dT

    def get_iv(self, log_moneyness: float, T: float) -> float:
        """Interpolated implied vol at arbitrary (log_moneyness, T)."""
        w, _ = self.get_total_variance_and_derivative(log_moneyness, T)
        return float(np.sqrt(w / max(T, 1e-8)))

    def get_iv_grid(self, moneyness_grid: np.ndarray,
                    T_grid: np.ndarray) -> np.ndarray:
        """Returns IV surface on a (len(T_grid), len(moneyness_grid)) grid."""
        return np.array([
            [self.get_iv(k, T) for k in moneyness_grid]
            for T in T_grid
        ])

    def check_calendar_arbitrage(self, n_points: int = 200) -> dict:
        """
        Static no-arbitrage check: calendar spread.

        A vol surface is calendar-spread arbitrage-free iff total variance
        w(k, T) is non-decreasing in T for every k.  Checks on a dense
        log-moneyness grid across all adjacent expiry pairs.

        Returns
        -------
        dict with keys:
            calendar_arbitrage_free : bool
            violations : list of (T_lo, T_hi, n_violations) tuples
        """
        k_grid = np.linspace(-0.30, 0.30, n_points)
        violations = []
        for i in range(len(self._T_vals) - 1):
            T_lo, T_hi = self._T_vals[i], self._T_vals[i + 1]
            w_lo = _svi_total_variance(k_grid, *self._slices[T_lo])
            w_hi = _svi_total_variance(k_grid, *self._slices[T_hi])
            n_viol = int(np.sum(w_lo > w_hi + 1e-8))
            if n_viol > 0:
                violations.append((float(T_lo), float(T_hi), n_viol))
        return {
            'calendar_arbitrage_free': len(violations) == 0,
            'violations': violations,
        }

    def check_butterfly_arbitrage(self, n_points: int = 200) -> dict:
        """
        Static no-arbitrage check: butterfly (convexity).

        Gatheral (2006): a smile slice is butterfly-arbitrage-free iff
            g(k) = (1 - k·w'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2 > 0

        for all k, where w' = ∂w/∂k and w'' = ∂²w/∂k².

        Returns
        -------
        dict with keys:
            butterfly_arbitrage_free : bool  (True iff ALL slices pass)
            fraction_valid           : float  fraction of (T, k) grid points where g > 0
            per_slice                : dict mapping T → fraction_valid for that slice
        """
        k_grid = np.linspace(-0.30, 0.30, n_points)
        per_slice: dict[float, float] = {}
        total_pts = 0
        valid_pts = 0

        for T_val, params in self._slices.items():
            w = _svi_total_variance(k_grid, *params)
            wk = _svi_dw_dk(k_grid, *params)
            wkk = _svi_d2w_dk2(k_grid, *params)

            A = 1.0 - k_grid * wk / (2.0 * np.maximum(w, 1e-12))
            g = A ** 2 - (wk ** 2 / 4.0) * (1.0 / np.maximum(w, 1e-12) + 0.25) + wkk / 2.0

            n_valid = int(np.sum(g > 0))
            per_slice[float(T_val)] = n_valid / n_points
            valid_pts += n_valid
            total_pts += n_points

        fraction = valid_pts / total_pts if total_pts > 0 else 0.0
        return {
            'butterfly_arbitrage_free': fraction == 1.0,
            'fraction_valid': round(fraction, 4),
            'per_slice': {round(T, 6): round(f, 4) for T, f in per_slice.items()},
        }
