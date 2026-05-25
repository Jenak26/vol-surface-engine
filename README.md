# Volatility Surface Engine

A derivatives pricing system built from first principles — from Black-Scholes to a fully calibrated Heston stochastic volatility model on live NSE (NIFTY) options data.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

| Phase | Models & Techniques |
|-------|-------------------|
| **Phase 1 — Black-Scholes** | Closed-form pricing, all 5 Greeks analytically + finite-difference numerical verification |
| **Phase 2 — Monte Carlo** | GBM path simulation, antithetic variates, control variates, Asian & barrier options |
| **Phase 3 — American Options** | Longstaff-Schwartz LSM with Laguerre regression, early exercise boundary, CRR binomial tree (both methods cross-validated) |
| **Phase 4 — Volatility Surface** | Newton-Raphson IV solver, SVI surface fitting, Heston Fourier pricing + calibration |

---

## Key Results

| Result | Value |
|--------|-------|
| BS pricer accuracy (vs analytical) | < 0.01 |
| MC convergence to BS (500k paths) | < 0.15 |
| Antithetic variance reduction | ~27% vs naive |
| Control variate variance reduction | ~40% vs naive |
| American put early exercise premium (deep ITM) | > 1.0 |
| IV solver round-trip error | < 1e-5 |
| Heston calibration RMSE | 0.02 vol points |

---

## Quick Start

```bash
git clone https://github.com/Jenak26/vol-surface-engine.git
cd vol-surface-engine
pip install -r requirements.txt
pytest -v                          # run all 72 tests
jupyter notebook notebooks/        # open the notebooks
```

---

## Project Structure

```
vol-surface-engine/
├── src/
│   ├── black_scholes.py       # BS price + 5 Greeks (analytical + numerical)
│   ├── monte_carlo.py         # GBM paths, European MC, variance reduction, exotics
│   ├── american_options.py    # Longstaff-Schwartz LSM + early exercise boundary
│   ├── binomial_tree.py       # CRR binomial tree for American options
│   ├── binomial_tree.py       # CRR binomial tree for American options
│   ├── implied_vol.py         # Newton-Raphson IV solver
│   ├── vol_surface.py         # SVI surface fitting + interpolation
│   └── heston.py              # Heston Fourier pricing + calibration
├── data/
│   ├── fetcher.py             # NSE options chain fetcher (live + synthetic fallback)
│   └── preprocessor.py        # Quote cleaning, mid price, IV computation
├── tests/                     # 72 tests across all modules
├── notebooks/
│   ├── 01_bs_greeks.ipynb     # Greeks surfaces, put-call parity
│   ├── 02_monte_carlo.ipynb   # Convergence plots, variance reduction, exotics
│   ├── 03_american_options.ipynb  # LSM vs binomial, early exercise boundary
│   └── 04_vol_surface.ipynb   # Smile plot, 3D surface, Heston calibration
└── requirements.txt
```

---

## Notebooks

### 01 — Black-Scholes Greeks
- Greeks (delta, gamma, vega, theta, rho) as functions of spot and time
- 3D Greek surfaces over (S, σ) grid
- Put-call parity verification

### 02 — Monte Carlo
- MC error vs number of paths (convergence)
- Sample GBM path visualisation
- Variance comparison: naive vs antithetic vs control variate (30-trial std dev)
- Asian vs European call price vs strike
- Down-and-out barrier price vs barrier level

### 03 — American Options
- American vs European put premium across spot range
- Early exercise boundary S*(t) over time
- LSM vs CRR binomial tree price comparison
- Convergence of binomial tree with n_steps

### 04 — Volatility Surface
- Volatility smile per expiry (BS flat-vol baseline overlay)
- Interactive 3D implied vol surface (Plotly)
- Heston calibration: κ, θ, σᵥ, ρ, v₀
- Heston fit vs market IV scatter

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| NumPy / SciPy | Numerical computing, integration, optimisation |
| Pandas | Options chain data handling |
| Matplotlib / Plotly | 2D charts + interactive 3D surface |
| nsepython | Live NSE (NIFTY) options chain data |
| pytest | 72-test suite across all 4 phases |

---

## Mathematics

**Black-Scholes:** `C = S·N(d₁) − K·e^(−rT)·N(d₂)`

**GBM:** `dS = rS dt + σS dW`  →  `S_T = S·exp((r − σ²/2)T + σ√T·Z)`

**SVI total variance:** `w(k) = a + b·[ρ(k−m) + √((k−m)² + σ²)]`

**Heston characteristic function** (Fourier inversion via Lewis 2001):  
Variance follows `dv = κ(θ−v)dt + σᵥ√v dW_v` with `corr(dW, dW_v) = ρ`

**LSM early exercise:** At each timestep, regress discounted continuation values onto Laguerre basis functions of the current stock price, exercise when intrinsic > continuation.

---

## License

MIT
