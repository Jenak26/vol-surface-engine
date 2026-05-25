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
    d = np.sqrt((rho * sigma_v * i * phi - b) ** 2
                - sigma_v ** 2 * (2 * u * i * phi - phi ** 2))
    g = (b - rho * sigma_v * i * phi + d) / (b - rho * sigma_v * i * phi - d)

    exp_dT = np.exp(d * T)
    C = r * i * phi * T + (a / sigma_v ** 2) * (
        (b - rho * sigma_v * i * phi + d) * T
        - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - rho * sigma_v * i * phi + d) / sigma_v ** 2) * (
        (1 - exp_dT) / (1 - g * exp_dT)
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


def calibrate_heston(log_moneyness: np.ndarray, market_ivs: np.ndarray,
                     T: float, S: float, r: float) -> tuple[dict, float]:
    """
    Calibrate Heston parameters to a single expiry slice.

    Minimises RMSE between Heston-implied vols and market implied vols.

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
    rmse  : root mean squared error in vol units
    """
    F = S * np.exp(r * T)
    strikes = F * np.exp(log_moneyness)

    def objective(x):
        kappa, theta, sigma_v, rho, v0 = x
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1 or v0 <= 0:
            return 1e6
        # Feller condition: 2*kappa*theta > sigma_v^2 ensures variance stays positive
        if 2 * kappa * theta < sigma_v ** 2:
            return 1e6
        errors = []
        for K_i, iv_mkt in zip(strikes, market_ivs):
            iv_h = heston_iv(S, K_i, T, r, kappa, theta, sigma_v, rho, v0)
            if iv_h is None:
                errors.append(0.10)  # penalise non-convergence
            else:
                errors.append(iv_h - iv_mkt)
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    atm_var = float(np.mean(market_ivs ** 2))
    x0 = [1.5, atm_var, 0.3, -0.5, atm_var]
    bounds = [(0.1, 10), (0.001, 1.0), (0.05, 2.0), (-0.99, 0.99), (0.001, 1.0)]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500})

    kappa, theta, sigma_v, rho, v0 = result.x
    params = {'kappa': float(kappa), 'theta': float(theta),
              'sigma_v': float(sigma_v), 'rho': float(rho), 'v0': float(v0)}
    return params, float(result.fun)
