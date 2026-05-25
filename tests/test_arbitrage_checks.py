import pytest
import numpy as np
from data.fetcher import _generate_synthetic_chain
from data.preprocessor import clean_option_chain
from src.vol_surface import VolatilitySurface


@pytest.fixture(scope='module')
def surface():
    df = _generate_synthetic_chain()
    spot = df['spot'].iloc[0]
    r = df['r'].iloc[0]
    cleaned = clean_option_chain(df, spot=spot, r=r, min_oi=0, min_volume=0)
    return VolatilitySurface(cleaned, spot=spot, r=r)


class TestCalendarArbitrage:
    def test_returns_expected_keys(self, surface):
        result = surface.check_calendar_arbitrage()
        assert 'calendar_arbitrage_free' in result
        assert 'violations' in result

    def test_result_is_bool(self, surface):
        result = surface.check_calendar_arbitrage()
        assert isinstance(result['calendar_arbitrage_free'], bool)

    def test_synthetic_surface_calendar_free(self, surface):
        # Synthetic data is generated with monotone total variance by construction
        result = surface.check_calendar_arbitrage()
        assert result['calendar_arbitrage_free'], (
            f"Unexpected calendar violations: {result['violations']}"
        )

    def test_violations_is_list(self, surface):
        result = surface.check_calendar_arbitrage()
        assert isinstance(result['violations'], list)


class TestButterflyArbitrage:
    def test_returns_expected_keys(self, surface):
        result = surface.check_butterfly_arbitrage()
        assert 'butterfly_arbitrage_free' in result
        assert 'fraction_valid' in result
        assert 'per_slice' in result

    def test_fraction_in_unit_interval(self, surface):
        result = surface.check_butterfly_arbitrage()
        assert 0.0 <= result['fraction_valid'] <= 1.0

    def test_synthetic_surface_mostly_butterfly_free(self, surface):
        # Smooth SVI fit on well-behaved data should be > 90% g > 0
        result = surface.check_butterfly_arbitrage()
        assert result['fraction_valid'] > 0.90, (
            f"Butterfly fraction too low: {result['fraction_valid']:.2%}"
        )

    def test_per_slice_fractions_in_range(self, surface):
        result = surface.check_butterfly_arbitrage()
        for T_val, frac in result['per_slice'].items():
            assert 0.0 <= frac <= 1.0, f"T={T_val}: fraction={frac}"
