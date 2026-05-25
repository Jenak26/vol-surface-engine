import numpy as np
from src.vol_surface import VolatilitySurface, _svi_total_variance, _svi_dw_dk, _svi_d2w_dk2


def _interpolated_svi_derivatives(surface: VolatilitySurface,
                                   k: float, T: float) -> tuple[float, float, float]:
    """
    Returns (w, ∂w/∂k, ∂²w/∂k²) at (k, T).

    Derivatives are computed analytically from the SVI parameters and
    interpolated in time using a natural cubic spline to match get_iv().
    """
    k_arr = np.array([k])
    T_vals = surface._T_vals
    T_clamped = max(min(T, T_vals[-1]), T_vals[0])

    if len(T_vals) < 3:
        # Fall back to linear interpolation if < 3 slices
        if T_clamped <= T_vals[0]:
            p = surface._slices[T_vals[0]]
            return (_svi_total_variance(k_arr, *p)[0],
                    _svi_dw_dk(k_arr, *p)[0],
                    _svi_d2w_dk2(k_arr, *p)[0])

        if T_clamped >= T_vals[-1]:
            p = surface._slices[T_vals[-1]]
            return (_svi_total_variance(k_arr, *p)[0],
                    _svi_dw_dk(k_arr, *p)[0],
                    _svi_d2w_dk2(k_arr, *p)[0])

        idx = int(np.searchsorted(T_vals, T_clamped)) - 1
        T_lo, T_hi = T_vals[idx], T_vals[idx + 1]
        alpha = (T_clamped - T_lo) / (T_hi - T_lo)

        p_lo, p_hi = surface._slices[T_lo], surface._slices[T_hi]

        w = ((1 - alpha) * _svi_total_variance(k_arr, *p_lo)[0]
             + alpha * _svi_total_variance(k_arr, *p_hi)[0])
        wk = ((1 - alpha) * _svi_dw_dk(k_arr, *p_lo)[0]
              + alpha * _svi_dw_dk(k_arr, *p_hi)[0])
        wkk = ((1 - alpha) * _svi_d2w_dk2(k_arr, *p_lo)[0]
               + alpha * _svi_d2w_dk2(k_arr, *p_hi)[0])
        return w, wk, wkk

    # Cubic spline interpolation in time
    from scipy.interpolate import CubicSpline
    w_list, wk_list, wkk_list = [], [], []
    for T_i in T_vals:
        p = surface._slices[T_i]
        w_list.append(_svi_total_variance(k_arr, *p)[0])
        wk_list.append(_svi_dw_dk(k_arr, *p)[0])
        wkk_list.append(_svi_d2w_dk2(k_arr, *p)[0])

    spline_w = CubicSpline(T_vals, w_list, bc_type='natural')
    spline_wk = CubicSpline(T_vals, wk_list, bc_type='natural')
    spline_wkk = CubicSpline(T_vals, wkk_list, bc_type='natural')

    return float(spline_w(T_clamped)), float(spline_wk(T_clamped)), float(spline_wkk(T_clamped))


def local_vol_from_svi(surface: VolatilitySurface,
                       log_moneyness: float, T: float) -> float:
    """
    Dupire local volatility extracted from the fitted SVI vol surface.

    Implements Gatheral (2006) "The Volatility Surface", eq. 1.7:

        σ²_loc(k, T) = (∂w/∂T) / g(k, T)

    where w = σ²_imp · T  is total implied variance,  k = log(K/F), and

        g = (1 − k·w_k / (2w))² − w_k² / 4 · (1/w + 1/4) + w_kk / 2

    Uses exact analytical time derivatives ∂w/∂T from the time-dimension
    cubic spline, completely avoiding numerical finite differences.
    """
    k = float(log_moneyness)
    T = float(T)

    # Get smooth total variance and its exact analytical time derivative
    w, w_T = surface.get_total_variance_and_derivative(k, T)

    if w_T <= 1e-12:
        return 0.0  # calendar-spread arbitrage at this point

    # analytical k-derivatives (interpolated between SVI slices via spline)
    w_interp, w_k, w_kk = _interpolated_svi_derivatives(surface, k, T)

    if w <= 1e-12:
        return 0.0

    # ── Gatheral butterfly g function (denominator) ───────────────────────────
    inv_w = 1.0 / w
    A = 1.0 - k * w_k * inv_w / 2.0
    g = A ** 2 - (w_k ** 2 / 4.0) * (inv_w + 0.25) + w_kk / 2.0

    if g <= 1e-12:
        return 0.0  # butterfly arbitrage at this point

    return float(np.sqrt(max(w_T / g, 0.0)))


def local_vol_grid(surface: VolatilitySurface,
                   moneyness_grid: np.ndarray,
                   T_grid: np.ndarray) -> np.ndarray:
    """
    Dupire local vol on a (len(T_grid), len(moneyness_grid)) grid.

    Returns σ_loc surface suitable for Matplotlib / Plotly visualisation.
    """
    return np.array([
        [local_vol_from_svi(surface, float(k), float(T)) for k in moneyness_grid]
        for T in T_grid
    ])
