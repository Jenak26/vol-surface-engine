# src/heston.py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from src.black_scholes import bs_price
from src.implied_vol import implied_vol


def _heston_cf_j(phi: complex, S: float, T: float, r: float,
                 kappa: float, theta: float, sigma_v: float,
                 rho: float, v0: float, j: int) -> complex:
    """
    Heston (1993) auxiliary characteristic functions for the two risk-neutral
    probabilities P1 (stock-numeraire) and P2 (money-market-numeraire).

    Uses the branch-cut-free stable formulation (Gatheral 2006, Albrecher 2007)
    which uses e^(-d*T) instead of e^(d*T) to prevent exponential overflow.

    j=1: uses u=+0.5, b=kappa-rho*sigma_v  (stock measure)
    j=2: uses u=-0.5, b=kappa               (risk-neutral / money-market measure)

    Returns the integrand CF evaluated at phi for probability Pj.
    """
    i = complex(0, 1)
    if j == 1:
        u = 0.5
        b = kappa - rho * sigma_v
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    d = np.sqrt((b - rho * sigma_v * i * phi) ** 2
                - sigma_v ** 2 * (2 * u * i * phi - phi ** 2))
    
    # Stable Gatheral formulation: g has -d in numerator, +d in denominator
    g = (b - rho * sigma_v * i * phi - d) / (b - rho * sigma_v * i * phi + d)

    # Use exp(-d*T) to avoid overflow (since Re(d) >= 0)
    exp_neg_dT = np.exp(-d * T)
    
    C = r * i * phi * T + (a / sigma_v ** 2) * (
        (b - rho * sigma_v * i * phi - d) * T
        - 2 * np.log((1 - g * exp_neg_dT) / (1 - g))
    )
    D = ((b - rho * sigma_v * i * phi - d) / sigma_v ** 2) * (
        (1 - exp_neg_dT) / (1 - g * exp_neg_dT)
    )
    return np.exp(C + D * v0 + i * phi * np.log(S))


def _heston_pj(j: int, S: float, K: float, T: float, r: float,
               kappa: float, theta: float, sigma_v: float,
               rho: float, v0: float) -> float:
    """
    Compute risk-neutral probability Pj via Fourier inversion.

    Pj = 1/2 + (1/pi) * integral_0^inf Re[exp(-i*phi*log(K)) * CF_j(phi) / (i*phi)] dphi
    """
    def integrand(phi: float) -> float:
        cf = _heston_cf_j(phi, S, T, r, kappa, theta, sigma_v, rho, v0, j)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    result, _ = quad(integrand, 1e-9, 200, limit=200)
    return 0.5 + result / np.pi


def heston_price(S: float, K: float, T: float, r: float,
                 kappa: float, theta: float, sigma_v: float,
                 rho: float, v0: float,
                 option_type: str = 'call',
                 n_integration_points: int = 100) -> float:
    """
    Heston (1993) call/put price via Fourier inversion of characteristic functions.

    C = S * P1 - K * exp(-r*T) * P2
    P = C - S + K * exp(-r*T)   (put-call parity)

    Parameters
    ----------
    S          : current spot price
    K          : strike price
    T          : time to expiry (years)
    r          : risk-free rate (continuously compounded)
    kappa      : mean-reversion speed of variance
    theta      : long-run variance (vol-of-vol squared of the stochastic vol)
    sigma_v    : vol-of-vol (volatility of the variance process)
    rho        : correlation between asset and variance Brownian motions
    v0         : initial variance
    option_type: 'call' or 'put'
    n_integration_points: passed for API compatibility; integration uses limit=200

    Returns
    -------
    float: option price
    """
    P1 = _heston_pj(1, S, K, T, r, kappa, theta, sigma_v, rho, v0)
    P2 = _heston_pj(2, S, K, T, r, kappa, theta, sigma_v, rho, v0)
    call_price = S * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        return float(max(call_price, 0.0))
    # Put via put-call parity
    put_price = call_price - S + K * np.exp(-r * T)
    return float(max(put_price, 0.0))


def heston_iv(S: float, K: float, T: float, r: float,
              kappa: float, theta: float, sigma_v: float,
              rho: float, v0: float) -> float | None:
    """Implied vol of a Heston option price (inverts BS to get IV)."""
    price = heston_price(S, K, T, r, kappa, theta, sigma_v, rho, v0, 'call')
    return implied_vol(price, S, K, T, r, 'call')


def heston_delta(S: float, K: float, T: float, r: float,
                 kappa: float, theta: float, sigma_v: float,
                 rho: float, v0: float,
                 option_type: str = 'call',
                 dS_pct: float = 0.005) -> float:
    """
    Heston delta via central finite difference (bump-and-reprice).
    dS_pct: bump size as fraction of spot (default 0.5%).
    """
    dS = S * dS_pct
    p_up = heston_price(S + dS, K, T, r, kappa, theta, sigma_v, rho, v0, option_type)
    p_dn = heston_price(S - dS, K, T, r, kappa, theta, sigma_v, rho, v0, option_type)
    return float((p_up - p_dn) / (2.0 * dS))


def heston_vega(S: float, K: float, T: float, r: float,
                kappa: float, theta: float, sigma_v: float,
                rho: float, v0: float,
                option_type: str = 'call',
                dv: float = 1e-4) -> float:
    """
    Heston vega: sensitivity to initial variance v0, via central bump.
    Note: this is ∂C/∂v0, not ∂C/∂σ_imp — divide by 2√v0 for the latter.
    """
    p_up = heston_price(S, K, T, r, kappa, theta, sigma_v, rho, v0 + dv, option_type)
    p_dn = heston_price(S, K, T, r, kappa, theta, sigma_v, rho, v0 - dv, option_type)
    return float((p_up - p_dn) / (2.0 * dv))


def heston_theta(S: float, K: float, T: float, r: float,
                 kappa: float, theta: float, sigma_v: float,
                 rho: float, v0: float,
                 option_type: str = 'call',
                 dt: float = 1.0 / 365.0) -> float:
    """
    Heston theta: daily price decay (negative = time decay).
    Uses a one-day forward difference: θ ≈ P(T - dt) - P(T).
    """
    if T <= dt:
        return 0.0
    p_now = heston_price(S, K, T, r, kappa, theta, sigma_v, rho, v0, option_type)
    p_later = heston_price(S, K, T - dt, r, kappa, theta, sigma_v, rho, v0, option_type)
    return float(p_later - p_now)


def calibrate_heston(log_moneyness: np.ndarray, market_ivs: np.ndarray,
                     T: float, S: float, r: float) -> tuple[dict, float]:
    """
    Calibrate Heston parameters to a single expiry slice.

    Uses Vega-weighted price errors as a fast proxy for volatility distance,
    completely eliminating the slow Newton-Raphson implied vol solver loop
    from the inner optimization.

    Parameters
    ----------
    log_moneyness : array of log(K/F) values where F = S*exp(r*T)
    market_ivs    : array of market implied volatilities (same length)
    T             : time to expiry (years)
    S             : current spot price
    r             : risk-free rate

    Returns
    -------
    params: dict with keys kappa, theta, sigma_v, rho, v0
    rmse  : root mean squared error in vol units (calculated at the optimal parameters)
    """
    from src.black_scholes import vega as bs_vega

    F = S * np.exp(r * T)
    strikes = F * np.exp(log_moneyness)

    # Pre-calculate market prices and Vegas to avoid redundant BS calls
    mkt_prices = []
    vegas = []
    for K_i, iv_mkt in zip(strikes, market_ivs):
        # Calibrate using call option prices
        p_mkt = bs_price(S, K_i, T, r, iv_mkt, 'call')
        # bs_vega() returns vega per 1% vol, multiply by 100 to get dC/dVol
        v_mkt = bs_vega(S, K_i, T, r, iv_mkt) * 100.0
        mkt_prices.append(p_mkt)
        vegas.append(max(v_mkt, 1e-4))  # floor vega to avoid division by zero

    def objective(x):
        kappa, theta, sigma_v, rho, v0 = x
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1 or v0 <= 0:
            return 1e6
        # Soft Feller condition: 2*kappa*theta > sigma_v^2 is desirable but often
        # violated in market fits. We allow it but can penalise if needed.
        # We keep the check here if Feller was required in the original code, but
        # relaxed so it doesn't penalise reasonable fits.
        # Let's keep the original check for compatibility:
        if 2 * kappa * theta < sigma_v ** 2:
            return 1e6
            
        errors = []
        for K_i, p_mkt, v_mkt in zip(strikes, mkt_prices, vegas):
            # Direct Fourier pricing under Heston (no IV root-finding)
            p_h = heston_price(S, K_i, T, r, kappa, theta, sigma_v, rho, v0, 'call')
            # Volatility error proxy: dVol ≈ dC / Vega
            errors.append((p_h - p_mkt) / v_mkt)
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    atm_var = float(np.mean(market_ivs ** 2))
    x0 = [1.5, atm_var, 0.3, -0.5, atm_var]
    bounds = [(0.1, 10), (0.001, 1.0), (0.05, 2.0), (-0.99, 0.99), (0.001, 1.0)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500})

    kappa, theta, sigma_v, rho, v0 = result.x
    params = {'kappa': float(kappa), 'theta': float(theta),
              'sigma_v': float(sigma_v), 'rho': float(rho), 'v0': float(v0)}
              
    # Compute the final true implied vol RMSE at the optimal parameters
    final_errors = []
    for K_i, iv_mkt in zip(strikes, market_ivs):
        iv_h = heston_iv(S, K_i, T, r, kappa, theta, sigma_v, rho, v0)
        if iv_h is None:
            final_errors.append(0.10)
        else:
            final_errors.append(iv_h - iv_mkt)
    final_rmse = float(np.sqrt(np.mean(np.array(final_errors) ** 2)))

    return params, final_rmse
