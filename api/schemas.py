from pydantic import BaseModel, Field
from typing import Literal, Optional


class PriceRequest(BaseModel):
    S: float = Field(..., gt=0, description="Spot price")
    K: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, description="Time to expiry (years)")
    r: float = Field(..., description="Continuously compounded risk-free rate")
    sigma: Optional[float] = Field(None, gt=0,
                                   description="Volatility — required for bs / mc / crr / lsm")
    option_type: Literal["call", "put"] = "call"
    model: Literal["bs", "mc", "mc_fast", "heston", "crr", "lsm"] = "bs"
    # Heston stochastic-vol parameters (required when model='heston')
    kappa: Optional[float] = Field(None, gt=0, description="Mean-reversion speed κ")
    theta: Optional[float] = Field(None, gt=0, description="Long-run variance θ")
    sigma_v: Optional[float] = Field(None, gt=0, description="Vol-of-vol σ_v")
    rho_v: Optional[float] = Field(None, ge=-1.0, le=1.0,
                                    description="Asset-vol Brownian correlation ρ")
    v0: Optional[float] = Field(None, gt=0, description="Initial variance v₀")
    # Monte Carlo knobs
    n_sims: int = Field(100_000, gt=0, description="Number of simulation paths")
    seed: int = Field(42, description="RNG seed (reproducibility)")


class PriceResponse(BaseModel):
    price: float
    model: str
    computation_ms: float


class GreeksRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    sigma: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"


class GreeksResponse(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class HestonGreeksRequest(BaseModel):
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    kappa: float = Field(..., gt=0)
    theta: float = Field(..., gt=0)
    sigma_v: float = Field(..., gt=0)
    rho_v: float = Field(..., ge=-1.0, le=1.0)
    v0: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"


class HestonGreeksResponse(BaseModel):
    delta: float
    vega: float
    theta: float


class IVRequest(BaseModel):
    market_price: float = Field(..., gt=0)
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0)
    r: float
    option_type: Literal["call", "put"] = "call"


class IVResponse(BaseModel):
    implied_vol: Optional[float]
    converged: bool
