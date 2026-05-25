import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestPriceBS:
    BASE = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.20}

    def test_call_price_positive(self):
        r = client.post("/price", json={**self.BASE, "option_type": "call", "model": "bs"})
        assert r.status_code == 200
        assert r.json()["price"] > 0

    def test_put_price_positive(self):
        r = client.post("/price", json={**self.BASE, "option_type": "put", "model": "bs"})
        assert r.status_code == 200
        assert r.json()["price"] > 0

    def test_put_call_parity(self):
        call = client.post("/price", json={**self.BASE, "option_type": "call", "model": "bs"}).json()["price"]
        put  = client.post("/price", json={**self.BASE, "option_type": "put",  "model": "bs"}).json()["price"]
        import math
        pcp = 100 - 100 * math.exp(-0.05 * 1.0)
        assert abs((call - put) - pcp) < 0.01

    def test_missing_sigma_returns_422(self):
        payload = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "model": "bs"}
        r = client.post("/price", json=payload)
        assert r.status_code == 422

    def test_model_field_in_response(self):
        r = client.post("/price", json={**self.BASE, "model": "bs"})
        assert r.json()["model"] == "bs"

    def test_computation_ms_non_negative(self):
        r = client.post("/price", json={**self.BASE, "model": "bs"})
        assert r.json()["computation_ms"] >= 0


class TestPriceMC:
    BASE = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.20,
            "n_sims": 20_000, "seed": 42}

    def test_mc_price_close_to_bs(self):
        mc_price = client.post("/price", json={**self.BASE, "model": "mc"}).json()["price"]
        bs_price = client.post("/price", json={**self.BASE, "model": "bs"}).json()["price"]
        assert abs(mc_price - bs_price) < 0.30


class TestPriceHeston:
    HESTON = {
        "S": 100, "K": 100, "T": 1.0, "r": 0.05,
        "model": "heston", "option_type": "call",
        "kappa": 1.5, "theta": 0.04, "sigma_v": 0.3, "rho_v": -0.5, "v0": 0.04,
    }

    def test_heston_price_positive(self):
        r = client.post("/price", json=self.HESTON)
        assert r.status_code == 200
        assert r.json()["price"] > 0

    def test_missing_heston_param_returns_422(self):
        payload = {k: v for k, v in self.HESTON.items() if k != "kappa"}
        r = client.post("/price", json=payload)
        assert r.status_code == 422


class TestGreeks:
    BASE = {"S": 100, "K": 100, "T": 1.0, "r": 0.05, "sigma": 0.20, "option_type": "call"}

    def test_delta_in_range(self):
        r = client.post("/greeks", json=self.BASE)
        assert 0 < r.json()["delta"] < 1

    def test_gamma_positive(self):
        r = client.post("/greeks", json=self.BASE)
        assert r.json()["gamma"] > 0

    def test_vega_positive(self):
        r = client.post("/greeks", json=self.BASE)
        assert r.json()["vega"] > 0

    def test_theta_negative(self):
        r = client.post("/greeks", json=self.BASE)
        assert r.json()["theta"] < 0

    def test_all_five_greeks_present(self):
        r = client.post("/greeks", json=self.BASE)
        body = r.json()
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            assert greek in body


class TestImpliedVol:
    def test_round_trip(self):
        # Price with BS, then invert back — should recover sigma within 1e-4
        sigma = 0.25
        r = client.post("/price", json={
            "S": 100, "K": 105, "T": 0.5, "r": 0.05, "sigma": sigma, "model": "bs"
        })
        price = r.json()["price"]
        iv_resp = client.post("/implied-vol", json={
            "market_price": price, "S": 100, "K": 105, "T": 0.5, "r": 0.05
        }).json()
        assert iv_resp["converged"]
        assert abs(iv_resp["implied_vol"] - sigma) < 1e-4

    def test_below_intrinsic_not_converged(self):
        r = client.post("/implied-vol", json={
            "market_price": 0.001, "S": 100, "K": 50, "T": 1.0, "r": 0.05, "option_type": "call"
        })
        assert not r.json()["converged"]
