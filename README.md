# Volatility Surface Engine

A production-grade derivatives pricing system built from first principles — from Black-Scholes to a fully calibrated Heston stochastic volatility model with Dupire local vol extraction, served over a REST API.

![CI](https://github.com/Jenak26/vol-surface-engine/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![Tests](https://img.shields.io/badge/tests-100%2B%20passing-brightgreen)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

| Phase | Models & Techniques |
|-------|-------------------|
| **1 — Black-Scholes** | Closed-form pricing · 5 Greeks analytically · finite-difference numerical verification |
| **2 — Monte Carlo** | GBM paths · antithetic variates · control variates · Asian & barrier options · **Numba parallel JIT** |
| **3 — American Options** | Longstaff-Schwartz LSM with Laguerre regression · early exercise boundary · CRR binomial tree |
| **4 — Volatility Surface** | Newton-Raphson IV solver · SVI per-slice calibration · Heston Fourier pricing + calibration · **Dupire local vol** · **butterfly & calendar arbitrage checks** |
| **5 — REST API** | FastAPI multi-model pricing · analytical + Heston Greeks · IV inversion · Swagger UI |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI REST API                        │
│  POST /price   POST /greeks   POST /greeks/heston            │
│  POST /implied-vol            GET  /health                   │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │      src/ library     │
        ├───────────────────────┤
        │ black_scholes.py      │  BS price + 5 analytical Greeks
        │ monte_carlo.py        │  GBM · variance reduction · Numba JIT
        │ american_options.py   │  LSM (Longstaff-Schwartz 2001)
        │ binomial_tree.py      │  CRR American/European
        │ implied_vol.py        │  Newton-Raphson IV solver
        │ vol_surface.py        │  SVI fitting · arbitrage checks
        │ heston.py             │  Fourier pricing · calibration · Greeks
        │ local_vol.py          │  Dupire local vol (Gatheral 2006)
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │      data/ pipeline   │
        ├───────────────────────┤
        │ fetcher.py            │  NSE live options chain (+ synthetic fallback)
        │ preprocessor.py       │  Quote cleaning · mid price · IV computation
        └───────────────────────┘
```

---

## Key Results

| Metric | Value |
|--------|-------|
| BS pricer accuracy (vs analytical) | < 0.01 |
| MC convergence to BS (500k paths) | < 0.15 |
| Antithetic variance reduction | ~27% vs naive |
| Control variate variance reduction | ~40% vs naive |
| American put early exercise premium (deep ITM) | > 1.0 |
| IV solver round-trip error | < 1e-5 |
| Heston calibration RMSE | < 0.02 vol points |
| Numba MC speedup (500k paths, 252 steps) | **4–8× vs numpy** |

---

## Quick Start

```bash
git clone https://github.com/Jenak26/vol-surface-engine.git
cd vol-surface-engine
pip install -r requirements.txt

# Run test suite (100+ tests across all modules)
pytest -v

# Start the REST API
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### Docker

```bash
docker build -t vol-surface-engine .
docker run -p 8000:8000 vol-surface-engine
```

---

## REST API

Interactive Swagger docs at **`/docs`**, ReDoc at **`/redoc`** (auto-generated).

### Price an option

```bash
# Black-Scholes European call
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"S":100,"K":100,"T":1.0,"r":0.05,"sigma":0.20,"model":"bs","option_type":"call"}' \
  | python -m json.tool
# → {"price": 10.450584, "model": "bs", "computation_ms": 0.041}

# Heston stochastic vol call
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"S":100,"K":100,"T":1.0,"r":0.05,"model":"heston","option_type":"call",
       "kappa":1.5,"theta":0.04,"sigma_v":0.3,"rho_v":-0.5,"v0":0.04}'

# Longstaff-Schwartz American put
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"S":100,"K":100,"T":1.0,"r":0.05,"sigma":0.20,"model":"lsm","option_type":"put","n_sims":50000}'
```

### Supported models

| `model` | Description |
|---------|-------------|
| `bs` | Black-Scholes closed form |
| `mc` | Monte Carlo (numpy) |
| `mc_fast` | Monte Carlo (Numba parallel JIT) |
| `heston` | Heston stochastic vol (Fourier inversion) |
| `crr` | CRR binomial tree (American put) |
| `lsm` | Longstaff-Schwartz Monte Carlo (American put) |

### Greeks

```bash
# Black-Scholes analytical Greeks
curl -s -X POST http://localhost:8000/greeks \
  -H "Content-Type: application/json" \
  -d '{"S":100,"K":100,"T":1.0,"r":0.05,"sigma":0.20,"option_type":"call"}'
# → {"delta":0.636831,"gamma":0.018763,"vega":0.375239,"theta":-0.017575,"rho":0.532325}

# Heston model Greeks (bump-and-reprice)
curl -s -X POST http://localhost:8000/greeks/heston \
  -H "Content-Type: application/json" \
  -d '{"S":100,"K":100,"T":1.0,"r":0.05,"kappa":1.5,"theta":0.04,
       "sigma_v":0.3,"rho_v":-0.5,"v0":0.04,"option_type":"call"}'
```

### Implied Volatility

```bash
curl -s -X POST http://localhost:8000/implied-vol \
  -H "Content-Type: application/json" \
  -d '{"market_price":10.45,"S":100,"K":100,"T":1.0,"r":0.05}'
# → {"implied_vol": 0.19996, "converged": true}
```

---

## Dupire Local Volatility

Extracts the **local volatility surface σ_loc(k, T)** from the fitted SVI implied vol surface using the Gatheral (2006) formula:

```
σ²_loc(k, T) = (∂w/∂T) / g(k, T)

g = (1 − k·w_k/2w)² − w_k²/4·(1/w + 1/4) + w_kk/2
```

where `w = σ²_imp·T` is total implied variance and `k = log(K/F)`.

```python
from src.vol_surface import VolatilitySurface
from src.local_vol import local_vol_from_svi, local_vol_grid

surface = VolatilitySurface(cleaned_df, spot=22000, r=0.065)

# Point query
lv = local_vol_from_svi(surface, log_moneyness=-0.05, T=30/365)

# Full grid for plotting
import numpy as np
k_grid = np.linspace(-0.20, 0.20, 50)
T_grid = np.array([30/365, 67/365, 95/365])
lv_surface = local_vol_grid(surface, k_grid, T_grid)   # shape (3, 50)
```

---

## Arbitrage Checks

```python
# Calendar spread (total variance must be non-decreasing in T)
print(surface.check_calendar_arbitrage())
# → {'calendar_arbitrage_free': True, 'violations': []}

# Butterfly (g function must be positive everywhere)
print(surface.check_butterfly_arbitrage())
# → {'butterfly_arbitrage_free': True, 'fraction_valid': 0.9980, 'per_slice': {...}}
```

---

## Performance Benchmark

Numba JIT kernel compiles once and parallelises across all CPU cores:

```
$ python -m benchmarks.bench_mc

====================================================================
Monte Carlo GBM Path Simulation — NumPy vs Numba
====================================================================
    n_sims    numpy (ms)    numba (ms)       speedup
--------------------------------------------------------------------
    10,000         12.3           3.1          4.0x
    50,000         58.7          12.4          4.7x
   100,000        115.2          22.8          5.1x
   500,000        576.1          89.3          6.4x
====================================================================
Note: 500k × 252 steps = 126 M path-steps per call.
```

---

## Project Structure

```
vol-surface-engine/
├── api/
│   ├── main.py            # FastAPI app — pricing, Greeks, IV endpoints
│   └── schemas.py         # Pydantic request/response models
├── src/
│   ├── black_scholes.py   # BS price + 5 Greeks (analytical + numerical)
│   ├── monte_carlo.py     # GBM paths · variance reduction · Numba JIT
│   ├── american_options.py# Longstaff-Schwartz LSM + early exercise boundary
│   ├── binomial_tree.py   # CRR binomial tree (American + European)
│   ├── implied_vol.py     # Newton-Raphson IV solver
│   ├── vol_surface.py     # SVI fitting · calendar/butterfly arbitrage checks
│   ├── heston.py          # Fourier pricing · calibration · Greeks
│   └── local_vol.py       # Dupire local vol extraction (Gatheral 2006)
├── data/
│   ├── fetcher.py         # NSE live options chain + synthetic fallback
│   └── preprocessor.py    # Quote cleaning · mid price · IV computation
├── tests/                 # 100+ tests: theory-grounded, not just smoke tests
├── benchmarks/
│   └── bench_mc.py        # NumPy vs Numba timing comparison
├── notebooks/
│   ├── 01_bs_greeks.ipynb
│   ├── 02_monte_carlo.ipynb
│   ├── 03_american_options.ipynb
│   └── 04_vol_surface.ipynb
├── Dockerfile
└── requirements.txt
```

---

## Mathematics

**Black-Scholes:** `C = S·N(d₁) − K·e^(−rT)·N(d₂)`

**GBM:** `dS = rS dt + σS dW`  →  `S_T = S·exp((r − σ²/2)T + σ√T·Z)`

**SVI total variance:** `w(k) = a + b·[ρ(k−m) + √((k−m)² + σ²)]`

**Dupire local variance** (Gatheral 2006):
```
σ²_loc(k,T) = ∂w/∂T / [(1 − k·w_k/2w)² − w_k²/4·(1/w+1/4) + w_kk/2]
```

**Heston characteristic function** (Fourier inversion):
Variance follows `dv = κ(θ−v)dt + σᵥ√v dW_v` with `corr(dW, dW_v) = ρ`

**LSM early exercise:** Regress discounted continuation values onto Laguerre basis functions; exercise when intrinsic > continuation.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| NumPy / SciPy | Numerical computing, integration, optimisation |
| Numba | JIT parallel compilation for MC path simulation |
| Pandas | Options chain data handling |
| FastAPI + Pydantic | REST API with auto-generated Swagger docs |
| Matplotlib / Plotly | 2D charts + interactive 3D surface |
| nsepython | Live NSE (NIFTY) options chain data |
| pytest + httpx | 100+ tests — pricing theory + API integration |

---

## License

MIT
