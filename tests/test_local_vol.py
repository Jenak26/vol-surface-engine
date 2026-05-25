import pytest
import numpy as np
from data.fetcher import _generate_synthetic_chain
from data.preprocessor import clean_option_chain
from src.vol_surface import VolatilitySurface
from src.local_vol import local_vol_from_svi, local_vol_grid


@pytest.fixture(scope='module')
def surface():
    df = _generate_synthetic_chain()
    spot = df['spot'].iloc[0]
    r = df['r'].iloc[0]
    cleaned = clean_option_chain(df, spot=spot, r=r, min_oi=0, min_volume=0)
    return VolatilitySurface(cleaned, spot=spot, r=r)


class TestDupireLocalVol:
    def test_atm_local_vol_positive(self, surface):
        lv = local_vol_from_svi(surface, log_moneyness=0.0, T=30/365)
        assert lv > 0.0

    def test_local_vol_reasonable_range(self, surface):
        lv = local_vol_from_svi(surface, log_moneyness=0.0, T=30/365)
        assert 0.05 < lv < 2.0, f"ATM local vol out of expected range: {lv:.4f}"

    def test_otm_put_wing_elevated(self, surface):
        # Negative skew in synthetic data → put-wing local vol > ATM local vol
        lv_atm = local_vol_from_svi(surface, log_moneyness=0.0,   T=30/365)
        lv_wing = local_vol_from_svi(surface, log_moneyness=-0.10, T=30/365)
        assert lv_wing > lv_atm, (
            f"Put wing ({lv_wing:.4f}) should exceed ATM ({lv_atm:.4f}) under negative skew"
        )

    def test_second_expiry_positive(self, surface):
        lv = local_vol_from_svi(surface, log_moneyness=0.0, T=67/365)
        assert lv > 0.0

    def test_local_vol_non_negative_across_atm_grid(self, surface):
        for k in np.linspace(-0.10, 0.10, 15):
            lv = local_vol_from_svi(surface, k, 30/365)
            assert lv >= 0.0, f"Negative local vol at k={k:.3f}: {lv}"

    def test_grid_shape(self, surface):
        k_grid = np.linspace(-0.15, 0.15, 5)
        T_grid = np.array([30/365, 67/365])
        grid = local_vol_grid(surface, k_grid, T_grid)
        assert grid.shape == (2, 5)

    def test_grid_non_negative(self, surface):
        k_grid = np.linspace(-0.10, 0.10, 8)
        T_grid = np.array([30/365, 67/365])
        grid = local_vol_grid(surface, k_grid, T_grid)
        assert np.all(grid >= 0.0)
