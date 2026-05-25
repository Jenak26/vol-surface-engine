"""
Options Pricing REST API
========================
Run locally:
    uvicorn api.main:app --reload

Interactive docs:
    http://localhost:8000/docs
"""
import time
import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import (
    PriceRequest, PriceResponse,
    GreeksRequest, GreeksResponse,
    HestonGreeksRequest, HestonGreeksResponse,
    IVRequest, IVResponse,
)
from src.black_scholes import bs_price, delta, gamma, vega, theta, rho
from src.monte_carlo import mc_european, simulate_gbm_paths_fast
from src.heston import heston_price, heston_delta, heston_vega, heston_theta
from src.binomial_tree import crr_american_put
from src.american_options import lsm_american_put
from src.implied_vol import implied_vol

app = FastAPI(
    title="Options Pricing API",
    description=(
        "Multi-model derivatives pricing engine.\n\n"
        "**Models:** Black-Scholes · Monte Carlo (plain + Numba-parallel) · "
        "Heston stochastic vol · Longstaff-Schwartz American · CRR Binomial tree\n\n"
        "**Greeks:** BS analytical · Heston bump-and-reprice\n\n"
        "**Utilities:** Newton-Raphson implied-vol inversion"
    ),
    version="1.0.0",
    contact={"name": "Vol Surface Engine"},
)


from fastapi.responses import HTMLResponse

# ── Meta ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def read_root():
    """Return the premium quantitative pricing dashboard homepage."""
    with open("api/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health", tags=["Meta"])
def health():
    """Liveness probe."""
    return {"status": "ok", "version": app.version}


# ── Pricing ──────────────────────────────────────────────────────────────────

@app.post("/price", response_model=PriceResponse, tags=["Pricing"])
def price_option(req: PriceRequest):
    """
    Price a European or American option using the requested model.

    | model    | description                                     |
    |----------|-------------------------------------------------|
    | `bs`     | Black-Scholes closed form                       |
    | `mc`     | Monte Carlo (numpy, antithetic-ready)           |
    | `mc_fast`| Monte Carlo with Numba parallel JIT             |
    | `heston` | Heston stochastic vol (Fourier inversion)       |
    | `crr`    | Cox-Ross-Rubinstein binomial tree (American put)|
    | `lsm`    | Longstaff-Schwartz LSM (American put)           |
    """
    t0 = time.perf_counter()
    price = _dispatch_price(req)
    ms = (time.perf_counter() - t0) * 1000
    return PriceResponse(price=round(float(price), 6),
                         model=req.model,
                         computation_ms=round(ms, 3))


def _dispatch_price(req: PriceRequest) -> float:
    try:
        if req.model == "bs":
            _need(req.sigma, "sigma", req.model)
            return bs_price(req.S, req.K, req.T, req.r, req.sigma, req.option_type)

        if req.model == "mc":
            _need(req.sigma, "sigma", req.model)
            return mc_european(req.S, req.K, req.T, req.r, req.sigma,
                               req.option_type, req.n_sims, req.seed)

        if req.model == "mc_fast":
            _need(req.sigma, "sigma", req.model)
            paths = simulate_gbm_paths_fast(req.S, req.T, req.r, req.sigma,
                                            n_sims=req.n_sims, n_steps=1, seed=req.seed)
            ST = paths[:, -1]
            if req.option_type == "call":
                payoffs = np.maximum(ST - req.K, 0.0)
            else:
                payoffs = np.maximum(req.K - ST, 0.0)
            return float(np.exp(-req.r * req.T) * np.mean(payoffs))

        if req.model == "heston":
            _need_heston(req)
            return heston_price(req.S, req.K, req.T, req.r,
                                req.kappa, req.theta, req.sigma_v, req.rho_v, req.v0,
                                req.option_type)

        if req.model == "crr":
            _need(req.sigma, "sigma", req.model)
            return crr_american_put(req.S, req.K, req.T, req.r, req.sigma)

        if req.model == "lsm":
            _need(req.sigma, "sigma", req.model)
            return lsm_american_put(req.S, req.K, req.T, req.r, req.sigma,
                                    req.n_sims, 50, req.seed)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc

    raise HTTPException(400, detail=f"Unknown model: {req.model}")


# ── Greeks ───────────────────────────────────────────────────────────────────

@app.post("/greeks", response_model=GreeksResponse, tags=["Greeks"])
def compute_greeks(req: GreeksRequest):
    """Black-Scholes analytical Greeks (Δ, Γ, ν, Θ, ρ)."""
    return GreeksResponse(
        delta=round(delta(req.S, req.K, req.T, req.r, req.sigma, req.option_type), 6),
        gamma=round(gamma(req.S, req.K, req.T, req.r, req.sigma), 6),
        vega=round(vega(req.S, req.K, req.T, req.r, req.sigma), 6),
        theta=round(theta(req.S, req.K, req.T, req.r, req.sigma, req.option_type), 6),
        rho=round(rho(req.S, req.K, req.T, req.r, req.sigma, req.option_type), 6),
    )


@app.post("/greeks/heston", response_model=HestonGreeksResponse, tags=["Greeks"])
def compute_heston_greeks(req: HestonGreeksRequest):
    """
    Heston model Greeks via bump-and-reprice finite differences.

    - **delta**: ∂C/∂S  (central difference, 0.5 % bump)
    - **vega**: ∂C/∂v₀  (sensitivity to initial variance)
    - **theta**: daily time decay P(T−1d) − P(T)
    """
    args = (req.S, req.K, req.T, req.r,
            req.kappa, req.theta, req.sigma_v, req.rho_v, req.v0, req.option_type)
    return HestonGreeksResponse(
        delta=round(heston_delta(*args), 6),
        vega=round(heston_vega(*args), 6),
        theta=round(heston_theta(*args), 6),
    )


# ── Implied Volatility ────────────────────────────────────────────────────────

@app.post("/implied-vol", response_model=IVResponse, tags=["Pricing"])
def compute_iv(req: IVRequest):
    """
    Implied volatility via Newton-Raphson inversion of the BS formula.
    Returns `null` and `converged=false` for options below intrinsic value
    or where vega is numerically zero (deep OTM / very short-dated).
    """
    iv = implied_vol(req.market_price, req.S, req.K, req.T, req.r, req.option_type)
    return IVResponse(
        implied_vol=round(iv, 8) if iv is not None else None,
        converged=iv is not None,
    )


# ── Validation helpers ────────────────────────────────────────────────────────

def _need(val, name: str, model: str):
    if val is None:
        raise HTTPException(422, detail=f"'{name}' is required for model='{model}'")


def _need_heston(req: PriceRequest):
    for field in ("kappa", "theta", "sigma_v", "rho_v", "v0"):
        if getattr(req, field) is None:
            raise HTTPException(422, detail=f"'{field}' is required for model='heston'")
