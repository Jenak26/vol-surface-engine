# tests/test_vol_surface.py
import pytest
import numpy as np
import pandas as pd
from data.fetcher import _generate_synthetic_chain
from data.preprocessor import clean_option_chain
from src.vol_surface import VolatilitySurface


class TestVolatilitySurface:
    @pytest.fixture(scope='class')
    def surface(self):
        df = _generate_synthetic_chain()
        spot = df['spot'].iloc[0]
        r    = df['r'].iloc[0]
        cleaned = clean_option_chain(df, spot=spot, r=r, min_oi=0, min_volume=0)
        return VolatilitySurface(cleaned, spot=spot, r=r)

    def test_surface_fits_without_error(self, surface):
        assert surface is not None

    def test_surface_interpolates_atm(self, surface):
        iv = surface.get_iv(log_moneyness=0.0, T=30/365)
        assert 0.05 < iv < 1.0

    def test_surface_shows_smile(self, surface):
        # OTM put wing (negative log-moneyness) should have higher vol than ATM
        iv_atm  = surface.get_iv(log_moneyness=0.0,   T=30/365)
        iv_wing = surface.get_iv(log_moneyness=-0.10, T=30/365)
        assert iv_wing > iv_atm

    def test_surface_interpolates_between_expiries(self, surface):
        # Should be able to interpolate at a T between known expiries
        iv = surface.get_iv(log_moneyness=0.0, T=50/365)
        assert 0.05 < iv < 1.0

    def test_surface_grid_returns_correct_shape(self, surface):
        moneyness_grid = np.linspace(-0.2, 0.2, 10)
        T_grid = np.array([30/365, 67/365])
        grid = surface.get_iv_grid(moneyness_grid, T_grid)
        assert grid.shape == (2, 10)
