import pytest
import numpy as np
from src.implied_vol import implied_vol
from src.black_scholes import bs_price
from data.fetcher import fetch_nse_option_chain, _generate_synthetic_chain
from data.preprocessor import clean_option_chain

S, K, T, r = 100.0, 100.0, 1.0, 0.05

class TestImpliedVol:
    def test_round_trip_atm(self):
        # Generate a BS price at sigma=0.25, recover sigma via IV solver
        true_sigma = 0.25
        market_price = bs_price(S, K, T, r, true_sigma, 'call')
        iv = implied_vol(market_price, S, K, T, r, 'call')
        assert abs(iv - true_sigma) < 1e-5

    def test_round_trip_itm_call(self):
        true_sigma = 0.30
        market_price = bs_price(120, 100, T, r, true_sigma, 'call')
        iv = implied_vol(market_price, 120, 100, T, r, 'call')
        assert abs(iv - true_sigma) < 1e-5

    def test_round_trip_otm_put(self):
        true_sigma = 0.22
        market_price = bs_price(S, 110, T, r, true_sigma, 'put')
        iv = implied_vol(market_price, S, 110, T, r, 'put')
        assert abs(iv - true_sigma) < 1e-5

    def test_round_trip_various_vols(self):
        for true_sigma in [0.10, 0.20, 0.40, 0.60, 0.80]:
            market_price = bs_price(S, K, T, r, true_sigma, 'call')
            iv = implied_vol(market_price, S, K, T, r, 'call')
            assert abs(iv - true_sigma) < 1e-4, f"Failed at sigma={true_sigma}"

    def test_deep_otm_returns_nan_or_value(self):
        # Very small price for deep OTM — solver may or may not converge
        # Just verify it doesn't crash
        result = implied_vol(0.0001, S, 200, T, r, 'call')
        assert result is None or (isinstance(result, float) and (np.isnan(result) or result > 0))

    def test_implied_vol_increases_with_market_price(self):
        price_low  = bs_price(S, K, T, r, 0.20, 'call')
        price_high = bs_price(S, K, T, r, 0.40, 'call')
        iv_low  = implied_vol(price_low,  S, K, T, r, 'call')
        iv_high = implied_vol(price_high, S, K, T, r, 'call')
        assert iv_high > iv_low


class TestDataPipeline:
    def test_synthetic_chain_has_expected_columns(self):
        df = _generate_synthetic_chain()
        for col in ['strike', 'expiry', 'option_type', 'last_price', 'iv', 'spot', 'T', 'r']:
            assert col in df.columns, f"Missing column: {col}"

    def test_synthetic_chain_nonempty(self):
        df = _generate_synthetic_chain()
        assert len(df) > 0

    def test_clean_removes_zero_price_rows(self):
        df = _generate_synthetic_chain()
        spot = df['spot'].iloc[0]
        r    = df['r'].iloc[0]
        df.loc[0, 'last_price'] = 0.0
        df.loc[0, 'bid'] = 0.0
        df.loc[0, 'ask'] = 0.0
        cleaned = clean_option_chain(df, spot=spot, r=r, min_oi=0, min_volume=0)
        assert len(cleaned) < len(df)

    def test_clean_adds_computed_iv(self):
        df = _generate_synthetic_chain()
        spot = df['spot'].iloc[0]
        r    = df['r'].iloc[0]
        cleaned = clean_option_chain(df, spot=spot, r=r, min_oi=0, min_volume=0)
        assert 'computed_iv' in cleaned.columns
        assert cleaned['computed_iv'].notna().all()
