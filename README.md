# Volatility Surface Engine

A production-grade derivatives pricing, calibration, and risk engine built from first principles — from Black-Scholes to a calibrated Heston stochastic volatility model and Dupire local volatility extraction. 

Features an interactive, glassmorphic dark-mode web dashboard deployed live on the cloud.

*   **Live Web Dashboard:** [FastAPI options dashboard on Hugging Face Spaces](https://huggingface.co/spaces/Jenak26/vol-surface-engine)
*   **Interactive API documentation:** `/docs` (Swagger UI) is available on the live web app.

---

## What It Does

| Component | Math & Models | Features |
|---|---|---|
| **1 — Pricing Models** | Black-Scholes · Heston Fourier Inversion · Longstaff-Schwartz Monte Carlo (LSM) · Binomial Tree (CRR) | European & American options · JIT-compiled MC path generation (Numba) |
| **2 — Greeks Engine** | Black-Scholes analytical · Heston bump-and-reprice finite differences | First-order ($\Delta, \nu, \Theta, \rho$) and second-order ($\Gamma$) sensitivities |
| **3 — Volatility Surface** | SVI (Stochastic Volatility Inspired) parametrisation · natural cubic spline interpolation | Butterfly arbitrage penalty functions · $C^2$ continuous time interpolation |
| **4 — Local Volatility** | Dupire local volatility extraction (Gatheral 2006) | Exact analytical time derivatives ($\partial w / \partial T$) via cubic splines (no finite difference noise) |
| **5 — REST API & UI** | FastAPI · Pydantic · Plotly.js | Premium responsive quantitative dashboard homepage (dark-mode glassmorphic interface) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Web Dashboard / REST API               │
│         GET  / (Home)           POST /price                 │
│         POST /implied-vol       POST /greeks                │
│         POST /greeks/heston     GET  /health                │
│         GET  /docs (Swagger)                                │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │      src/ library     │
        ├───────────────────────┤
        │ black_scholes.py      │  BS price + 5 analytical Greeks
        │ heston.py             │  Stable characteristic function · Fast calibration
        │ local_vol.py          │  Dupire local vol · analytical spline derivatives
        │ vol_surface.py        │  SVI fitting · butterfly arbitrage checks
        │ american_options.py   │  Longstaff-Schwartz LSM (American put)
        │ binomial_tree.py      │  CRR American/European tree
        │ implied_vol.py        │  Newton-Raphson IV solver
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

## Quantitative Theory & Mathematics

### 1. Implied Volatility Surface: SVI Fit
We parameterise the total implied variance $w(k, T) = \sigma^2_{imp}(k, T) \cdot T$ for each expiry slice using the SVI (Stochastic Volatility Inspired) model (Gatheral 2004):
$$w(k) = a + b \left[ \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right]$$
where $k = \ln(K/F)$ is the log-moneyness.
To guarantee that each slice is butterfly arbitrage-free, the calibration objective function enforces that the Gatheral butterfly density function $g(k)$ remains strictly positive:
$$g(k) = \left(1 - \frac{k w'(k)}{2w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} > 0$$
If $g(k) \le 0$ on a dense log-moneyness grid, a continuous penalty term is added to the least-squares fitting objective function to guide the L-BFGS-B optimizer toward arbitrage-free parameter sets.

### 2. Dupire Local Volatility Surface
Dupire's local variance $\sigma^2_{loc}(k, T)$ is extracted from the volatility surface using Gatheral's total variance representation:
$$\sigma^2_{loc}(k, T) = \frac{\frac{\partial w}{\partial T}}{\left(1 - \frac{k \frac{\partial w}{\partial k}}{2w}\right)^2 - \frac{\left(\frac{\partial w}{\partial k}\right)^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{\partial^2 w}{\partial k^2}}$$
*   **The Upgrade:** Instead of using noisy numerical finite differences to approximate the time derivative $\partial w/\partial T$ (which leads to step-discontinuities at expiry boundaries), the engine fits a natural cubic spline through total variances in the time dimension. The exact analytical derivative $\partial w/\partial T$ is computed directly from the spline, yielding a $C^2$ continuous, physically smooth local volatility grid.

### 3. Stable Heston Stochastic Volatility Model
The Heston model assumes the asset price $S_t$ and its variance $v_t$ follow the stochastic differential equations (SDEs):
$$dS_t = r S_t dt + \sqrt{v_t} S_t dW^1_t$$
$$dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW^2_t$$
with $d\langle W^1, W^2 \rangle_t = \rho dt$.
*   **Stable Characteristic Function:** The Fourier pricing kernel uses the branch-cut-free characteristic function (Albrecher et al. 2007) that utilizes $e^{-dT}$ to prevent numerical overflow for long maturities ($T \ge 10.0$ years):
    $$C(T) = r i \phi T + \frac{\kappa \theta}{\sigma_v^2} \left[ (\kappa - \rho \sigma_v i \phi - d) T - 2 \ln \left( \frac{1 - g e^{-dT}}{1 - g} \right) \right]$$
    $$D(T) = \frac{\kappa - \rho \sigma_v i \phi - d}{\sigma_v^2} \left[ \frac{1 - e^{-dT}}{1 - g e^{-dT}} \right]$$
    where $g = \frac{\kappa - \rho \sigma_v i \phi - d}{\kappa - \rho \sigma_v i \phi + d}$.
*   **Fast Vega-Weighted Calibration:** Instead of solving a nested Newton-Raphson implied volatility root-finding loop inside the optimizer (which is computationally expensive), the market IVs are converted to call prices and Vegas *once* before the calibration begins. The optimizer minimizes the Vega-weighted price errors:
    $$\text{Error} = \sum_i \left( \frac{C_{\text{heston}} - C_{\text{market}}}{\text{Vega}_{\text{market}}} \right)^2$$
    This acts as a first-order approximation of the implied volatility distance ($\Delta \sigma \approx \Delta C / \text{Vega}$), accelerating calibration speed by **10x–50x** while maintaining numerical consistency.

---

## Project Structure

*   **`api/`**
    *   `main.py`: FastAPI endpoints. Renders the interactive dashboard homepage at `/` and Swagger UI at `/docs`.
    *   `templates/index.html`: Dark-mode quantitative dashboard featuring input parameter calculators, card outputs, and Plotly 3D volatility surface visualizations.
*   **`src/`**
    *   `black_scholes.py`: Black-Scholes closed-form pricing and analytical/numerical Greeks.
    *   `heston.py`: Stable Heston Fourier pricing and fast Vega-weighted calibration.
    *   `local_vol.py`: Dupire local volatility extraction using analytical cubic spline derivatives.
    *   `vol_surface.py`: SVI implied volatility surface fitting with butterfly arbitrage penalties.
    *   `american_options.py`: Longstaff-Schwartz American put pricing and early exercise boundary extraction.
    *   `binomial_tree.py`: Cox-Ross-Rubinstein binomial tree for American/European options.
    *   `implied_vol.py`: Newton-Raphson implied volatility root-finder.
*   **`data/`**
    *   `fetcher.py`: Live NSE options chain data downloader with synthetic skew fallback.
    *   `preprocessor.py`: Raw quote cleaning, mid-price calculation, and IV inversion.
*   **`notebooks/`** (Descriptively labeled workspace files)
    *   `01_BlackScholes_Greeks.ipynb`: Greeks analysis (analytical vs numerical differences).
    *   `02_MonteCarlo_VarianceReduction_Exotics.ipynb`: JIT-compiled Monte Carlo pricing for exotic options.
    *   `03_AmericanPut_BinomialTree_vs_LSM.ipynb`: American options comparison.
    *   `04_SVI_Surface_DupireLocalVol_HestonCalibration.ipynb`: Surface calibration and extraction workflow.

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Jenak26/vol-surface-engine.git
cd vol-surface-engine
pip install -r requirements.txt
```

### 2. Run unit tests (122 passing tests verifying pricing theory and calibration)
```bash
pytest -v
```

### 3. Run the quantitative demonstration pipeline
This will fetch options quotes, perform SVI surface fitting, run the Heston calibration, extract the Dupire local volatility surface, and save the visualization plots:
```bash
python demo.py
```
This generates two plots: `volatility_surfaces.png` (Implied Vol vs Dupire Local Vol) and `heston_calibration_fit.png` (calibrated Heston smile vs market smile).

### 4. Run the Web Dashboard locally
```bash
uvicorn api.main:app --reload
```
Navigate to `http://localhost:8000` to view the interactive dashboard, or `http://localhost:8000/docs` to view the Swagger API documentation.

### 5. Docker Deployment
Build and run the Docker container:
```bash
docker build -t vol-surface-engine .
docker run -p 7860:7860 vol-surface-engine
```
Navigate to `http://localhost:7860` to access the running app.

---

## License

This project is licensed under the MIT License.
