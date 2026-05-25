import pytest
import numpy as np
from src.heston import heston_price, heston_iv, calibrate_heston
from src.black_scholes import bs_price

S, K, T, r = 100.0, 100.0, 1.0, 0.05

# Heston parameters that reproduce ~20% flat vol (should match BS closely)
FLAT_PARAMS = {'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.01,
               'rho': 0.0, 'v0': 0.04}

class TestHestonPricing:
    def test_heston_close_to_bs_when_flat(self):
        # With sigma_v≈0 and rho=0, Heston ≈ BS (vol ≈ constant at sqrt(theta))
        h_price = heston_price(S, K, T, r, **FLAT_PARAMS, option_type='call')
        bs = bs_price(S, K, T, r, 0.20, 'call')  # sqrt(0.04) = 0.20
        assert abs(h_price - bs) < 1e-3  # tight tolerance made possible by stable branch-cut-free characteristic function

    def test_heston_call_positive(self):
        params = {'kappa': 1.5, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.5, 'v0': 0.04}
        assert heston_price(S, K, T, r, **params, option_type='call') > 0

    def test_heston_put_positive(self):
        params = {'kappa': 1.5, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.5, 'v0': 0.04}
        assert heston_price(S, K, T, r, **params, option_type='put') > 0

    def test_heston_put_call_parity(self):
        params = {'kappa': 1.5, 'theta': 0.04, 'sigma_v': 0.3, 'rho': -0.5, 'v0': 0.04}
        call = heston_price(S, K, T, r, **params, option_type='call')
        put  = heston_price(S, K, T, r, **params, option_type='put')
        pcp  = S - K * np.exp(-r * T)
        assert abs((call - put) - pcp) < 0.05

    def test_heston_iv_produces_smile(self):
        # Negative rho → implied vols higher for low strikes (put skew)
        params = {'kappa': 2.0, 'theta': 0.04, 'sigma_v': 0.5, 'rho': -0.7, 'v0': 0.04}
        iv_otm_put = heston_iv(S, 90,  T, r, **params)  # OTM put
        iv_atm     = heston_iv(S, 100, T, r, **params)  # ATM
        iv_otm_call = heston_iv(S, 110, T, r, **params) # OTM call
        # With negative rho, vol should be highest for lowest strike
        assert iv_otm_put > iv_atm

class TestHestonCalibration:
    def test_calibration_reduces_error(self):
        from data.fetcher import _generate_synthetic_chain
        from data.preprocessor import clean_option_chain
        df = _generate_synthetic_chain()
        spot = df['spot'].iloc[0]
        r_rate = df['r'].iloc[0]
        cleaned = clean_option_chain(df, spot=spot, r=r_rate, min_oi=0, min_volume=0)
        # Use only near-term expiry for speed
        T_near = cleaned['T'].min()
        slice_df = cleaned[cleaned['T'] == T_near].head(10)

        params, rmse = calibrate_heston(
            slice_df['log_moneyness'].values,
            slice_df['computed_iv'].values,
            T_near, spot, r_rate
        )
        assert rmse < 0.10  # RMSE < 10 vol points
        assert params['kappa'] > 0
        assert params['theta'] > 0
        assert params['sigma_v'] > 0
        assert -1 < params['rho'] < 1
        assert params['v0'] > 0
