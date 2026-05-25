import numpy as np
from src.vol_surface import VolatilitySurface, _svi_total_variance, _svi_dw_dk, _svi_d2w_dk2


def _interpolated_svi_derivatives(surface: VolatilitySurface,
                                   k: float, T: float) -> tuple[float, float, float]:
    """
    Returns (w, ∂w/∂k, ∂²w/∂k²) at (k, T).

    Derivatives are computed analytically from the SVI parameters and
    linearly interpolated (in total-variance space) between bracketing
    expiry slices, consistent with get_iv().
    """
    k_arr = np.array([k])
    T_vals = surface._T_vals

    if T <= T_vals[0]:
        p = surface._slices[T_vals[0]]
        return (_svi_total_variance(k_arr, *p)[0],
                _svi_dw_dk(k_arr, *p)[0],
                _svi_d2w_dk2(k_arr, *p)[0])

    if T >= T_vals[-1]:
        p = surface._slices[T_vals[-1]]
        return (_svi_total_variance(k_arr, *p)[0],
                _svi_dw_dk(k_arr, *p)[0],
                _svi_d2w_dk2(k_arr, *p)[0])

    idx = int(np.searchsorted(T_vals, T)) - 1
    T_lo, T_hi = T_vals[idx], T_vals[idx + 1]
    alpha = (T - T_lo) / (T_hi - T_lo)   # 0 at T_lo, 1 at T_hi

    p_lo, p_hi = surface._slices[T_lo], surface._slices[T_hi]

    w = ((1 - alpha) * _svi_total_variance(k_arr, *p_lo)[0]
         + alpha * _svi_total_variance(k_arr, *p_hi)[0])
    wk = ((1 - alpha) * _svi_dw_dk(k_arr, *p_lo)[0]
          + alpha * _svi_dw_dk(k_arr, *p_hi)[0])
    wkk = ((1 - alpha) * _svi_d2w_dk2(k_arr, *p_lo)[0]
           + alpha * _svi_d2w_dk2(k_arr, *p_hi)[0])
    return w, wk, wkk


def local_vol_from_svi(surface: VolatilitySurface,
                       log_moneyness: float, T: float) -> float:
    """
    Dupire local volatility extracted from the fitted SVI vol surface.

    Implements Gatheral (2006) "The Volatility Surface", eq. 1.7:

        σ²_loc(k, T) = (∂w/∂T) / g(k, T)

    where w = σ²_imp · T  is total implied variance,  k = log(K/F), and

        g = (1 − k·w_k / (2w))² − w_k² / 4 · (1/w + 1/4) + w_kk / 2

    g > 0 everywhere is the butterfly no-arbitrage condition; ∂w/∂T > 0
    is the calendar-spread no-arbitrage condition. Returns 0.0 when either
    is violated (degenerate point on the surface).

    Parameters
    ----------
    surface      : fitted VolatilitySurface
    log_moneyness: k = log(K/F)
    T            : time to expiry (years)

    Returns
    -------
    σ_loc (annualised), ≥ 0.
    """
    k = float(log_moneyness)
    T = float(T)

    # ── ∂w/∂T via symmetric finite difference ────────────────────────────────
    # Step size: 1 calendar day, clamped so T_dn stays positive.
    dt = max(min(1.0 / 365.0, T * 0.1), 1e-6)
    T_up = T + dt / 2.0
    T_dn = max(T - dt / 2.0, 1e-8)

    def _total_var(tt: float) -> float:
        iv = surface.get_iv(k, tt)
        return iv * iv * tt

    w_T = (_total_var(T_up) - _total_var(T_dn)) / (T_up - T_dn)

    if w_T <= 1e-12:
        return 0.0  # calendar-spread arbitrage at this point

    # ── analytical k-derivatives (interpolated between SVI slices) ───────────
    w, w_k, w_kk = _interpolated_svi_derivatives(surface, k, T)

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
