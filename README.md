# 📈 Volatility Surface Engine

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Numba JIT](https://img.shields.io/badge/Numba-JIT--Compiled-orange?style=for-the-badge&logo=python&logoColor=white)](https://numba.pydata.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Jenak26/vol-surface-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A production-grade options pricing, calibration, and risk engine designed from first principles. Features a stable Heston stochastic volatility Fourier solver, arbitrage-free SVI implied volatility surface fitting, and smooth Dupire local volatility extraction. 

Delivered with a premium glassmorphic dark-mode web dashboard containing real-time options analytics and interactive 3D visualizations.

*   **Live Web Demonstration:** [Interactive Options Dashboard (Hugging Face Spaces)](https://huggingface.co/spaces/Jenak26/vol-surface-engine)
*   **API Interactive Documentation:** Swagger UI accessible live at `/docs`

---

## 🎯 Value Proposition & Target Performance

This engine is engineered to bridge the gap between mathematical model design and high-performance financial systems. By restructuring numerical kernels and implementing advanced stabilization techniques, it achieves the following benchmarks:

*   ⚡ **Sub-Millisecond Option Pricing:** Core evaluation routines written in NumPy and Numba JIT-compiled arrays, pricing standard European/American options and stochastic paths in microseconds.
*   🚀 **10x–50x Heston Calibration Speedup:** Minimizes a fast **Vega-weighted price error proxy** ($\Delta\sigma \approx \Delta C / \text{Vega}$) outside the optimization loop, completely bypassing the expensive nested Newton-Raphson implied volatility solvers.
*   🌊 **Smooth $C^2$ Continuous Dupire Surface:** Employs **natural cubic spline interpolation** across the maturity dimension, yielding clean analytical time derivatives ($\partial w / \partial T$) that eliminate discretization spikes and vertical cliffs.
*   🛡️ **Arbitrage-Free SVI Surfaces:** Guarantees no butterfly or calendar arbitrage by incorporating continuous penalty functions in the L-BFGS-B optimization loops.

---

## 🏗️ Core Architecture & Pipeline

The diagram below represents the end-to-end data pipeline and request-response flow, from raw market options data to the interactive Plotly front-end:

```mermaid
flowchart TD
    %% Source Data
    subgraph Data Layer
        A[NSE Live Options Chain API] -->|JSON Response| C[Quote Cleaner & Preprocessor]
        B[Synthetic Skew Generator] -->|Offline Fallback| C
    end

    %% Preprocessing
    C -->|Filter Mid-Prices & Spreads| D[Newton-Raphson IV Solver]
    D -->|Implied Volatilities & Market Vega| E[Cleaned Options Dataset]

    %% Numerical Engines
    subgraph Quantitative Engine
        E -->|Least-Squares Optimization| F[SVI Volatility Surface Model]
        F -->|Arbitrage Constraint Penalty| F
        
        E -->|Vega-Weighted Price Errors| G[Heston Model Calibration]
        G -->|Stable Fourier Inversion CF| G
        
        F -->|Time Dimension Cubic Spline| H[Dupire Local Volatility Engine]
        H -->|Analytical Time Derivatives dw/dT| H
    end

    %% Web Delivery
    subgraph REST API & UI (FastAPI Container)
        F & G & H -->|REST Endpoints / JSON Data| I[FastAPI Backend Router]
        I -->|HTML Response| J[Glassmorphic Dark-Mode Dashboard]
        I -->|Interactive API Docs| K[Swagger UI /docs]
    end

    %% Visualization
    J -->|Plotly.js| L[3D Volatility Surfaces]
    J -->|Plotly.js| M[2D Option Smiles & Greeks]
```

---

## 💡 Engine Highlights & Technical Upgrades

> [!NOTE]
> **Numerical Stability: The "Little Heston Trap" Resolution**
> Original Heston pricing code frequently experiences exponential overflow and numerical instability due to branch cuts in the complex logarithm at long maturities ($T \ge 10$ years). This engine implements the stable, branch-cut-free representation (Gatheral 2006, Albrecher 2007) by restructuring the characteristic function using $e^{-dT}$ terms, preventing numerical overflow and stabilizing integration.

> [!TIP]
> **Vega-Weighted Calibration Efficiency**
> Desk-ready model calibration requires running in milliseconds. Instead of running a nested Newton root-finder to find model IVs inside the optimizer loop, we compute the Black-Scholes Vega of market quotes once. The optimizer then minimizes the Vega-weighted price difference, which acts as a first-order approximation of the implied volatility distance:
> $$\text{Error} = \sum_i \left( \frac{C_{\text{heston}} - C_{\text{market}}}{\text{Vega}_{\text{market}}} \right)^2$$

> [!IMPORTANT]
> **$C^2$ Continuous Dupire Local Volatility**
> Finite difference approximations of the total variance time derivative ($\partial w / \partial T$) create step-discontinuity vertical cliffs in local volatility surfaces. To solve this, this engine fits a natural cubic spline in the time dimension across maturities, evaluating the exact analytical time derivative to yield a smooth, noise-free local volatility grid ready for finite-difference PDE or MC solvers.

---

## 📐 Quantitative Theory & Mathematics

### 1. Implied Volatility Surface: SVI Parameterisation
We model the total implied variance $w(k, T) = \sigma^2_{\text{imp}}(k, T) \cdot T$ for each maturity slice using the Stochastic Volatility Inspired (SVI) model (Gatheral 2004):
$$w(k) = a + b \left[ \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right]$$
where:
*   $k = \ln(K/F)$ is the log-moneyness.
*   $\{a, b, \rho, m, \sigma\}$ are the slice SVI parameters.

To guarantee that the calibrated surface is free of butterfly arbitrage, we enforce that the Gatheral butterfly density $g(k)$ remains strictly positive on a dense log-moneyness grid:
$$g(k) = \left(1 - \frac{k w'(k)}{2w(k)}\right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} > 0$$
Violations are heavily penalized in the optimization loss function to ensure arbitrage-free slices.

### 2. Dupire Local Volatility Surface
Under the Dupire local volatility framework, local variance $\sigma^2_{\text{loc}}(k, T)$ is extracted from the SVI total variance surface $w(k, T)$:
$$\sigma^2_{\text{loc}}(k, T) = \frac{\frac{\partial w}{\partial T}}{\left(1 - \frac{k \frac{\partial w}{\partial k}}{2w}\right)^2 - \frac{\left(\frac{\partial w}{\partial k}\right)^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{\partial^2 w}{\partial k^2}}$$
We compute the log-moneyness derivatives $\frac{\partial w}{\partial k}$ and $\frac{\partial^2 w}{\partial k^2}$ analytically from the SVI parameters, and evaluate the time derivative $\frac{\partial w}{\partial T}$ analytically using natural cubic spline coefficients in the time dimension.

### 3. Stable Heston Stochastic Volatility Model
The Heston model models the asset price $S_t$ and variance $v_t$ via coupled SDEs:
$$dS_t = r S_t dt + \sqrt{v_t} S_t dW^1_t$$
$$dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW^2_t$$
where $d\langle W^1, W^2 \rangle_t = \rho dt$.

Options are priced using Fourier transform inversion of the stable, branch-cut-free characteristic function:
$$C_{\text{heston}}(S_0, K, T) = S_0 \Pi_1 - K e^{-r T} \Pi_2$$
where the characteristic function is written as:
$$\Phi(u) = \exp\left( C(T, u) + D(T, u) v_0 + i u \ln(S_0 e^{r T}) \right)$$
$$C(T, u) = r i u T + \frac{\kappa \theta}{\sigma_v^2} \left[ (\kappa - \rho \sigma_v i u - d) T - 2 \ln \left( \frac{1 - g e^{-dT}}{1 - g} \right) \right]$$
$$D(T, u) = \frac{\kappa - \rho \sigma_v i u - d}{\sigma_v^2} \left[ \frac{1 - e^{-dT}}{1 - g e^{-dT}} \right]$$
with $d = \sqrt{(\rho \sigma_v i u - \kappa)^2 + \sigma_v^2 (i u + u^2)}$ and $g = \frac{\kappa - \rho \sigma_v i u - d}{\kappa - \rho \sigma_v i u + d}$.

---

## 🛠️ Tech Stack & Project Directory

### Technology Stack
*   **Backend Framework:** FastAPI, Uvicorn (REST API routing, asynchronous lifecycle)
*   **Numerical & Computation:** NumPy, SciPy (Optimization & Interpolation), Numba (Parallel JIT Monte Carlo compilation)
*   **Frontend Dashboard:** Vanilla HTML5/CSS3 (custom neon-glassmorphic stylesheet), Plotly.js (interactive 3D surfaces and option smiles)
*   **DevOps & Testing:** Docker (containerized deployment), Pytest (unit testing), GitHub Actions (automated CI testing)

### Directory Tree
```
vol-surface-engine/
├── .github/                # CI/CD Workflows
│   └── workflows/
│       └── tests.yml       # GitHub Actions Pytest Suite runs
├── api/                    # Web Application Layer
│   ├── templates/
│   │   └── index.html      # Responsive glassmorphic frontend UI
│   ├── __init__.py
│   ├── main.py             # FastAPI routing and entrypoint
│   └── schemas.py          # Pydantic schema validation models
├── data/                   # Data Acquisition & Cleaning
│   ├── fetcher.py          # NSE live options data downloader
│   └── preprocessor.py     # Options chain quote cleaning & IV inversion
├── notebooks/              # Quantitative Research Chapters
│   ├── 01_BlackScholes_Greeks.ipynb
│   ├── 02_MonteCarlo_VarianceReduction_Exotics.ipynb
│   ├── 03_AmericanPut_BinomialTree_vs_LSM.ipynb
│   └── 04_SVI_Surface_DupireLocalVol_HestonCalibration.ipynb
├── src/                    # Quant Core Library
│   ├── american_options.py # Longstaff-Schwartz LSM Monte Carlo solver
│   ├── binomial_tree.py    # Cox-Ross-Rubinstein (CRR) American tree
│   ├── black_scholes.py    # Black-Scholes analytical pricer & Greeks
│   ├── heston.py           # Branch-cut-free Heston & Vega-weighted calibration
│   ├── implied_vol.py      # Newton-Raphson implied volatility calculator
│   ├── local_vol.py        # Dupire Local Volatility extraction with splines
│   └── vol_surface.py      # SVI fitting & butterfly arbitrage penalty fitting
├── tests/                  # Unit Test Suite
│   ├── test_api.py         # API endpoint validation
│   ├── test_heston.py      # Heston numerical pricing & calibration tests
│   └── ...
├── Dockerfile              # Multi-stage production container
├── demo.py                 # Single-command demonstration script
├── requirements.txt        # Production dependency specifications
└── README.md               # Repository documentation
```

---

## 🚀 Installation & Local Development

### 1. Prerequisites
Ensure you have Python 3.12+ and Git installed on your system.

### 2. Local Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Jenak26/vol-surface-engine.git
cd vol-surface-engine
pip install -r requirements.txt
```

### 3. Run the Unit Tests
Execute the test suite of 122 tests verifying pricing kernels, Greeks, SVI fits, and APIs:
```bash
pytest -v
```

### 4. Execute the Demonstration Pipeline
Run the quantitative demo script to fetch data, fit an SVI surface, calibrate Heston, extract Dupire local volatility, and verify mathematical convergence:
```bash
python demo.py
```

### 5. Run the FastAPI Server Locally
Start the server using Uvicorn with auto-reload:
```bash
uvicorn api.main:app --reload
```
Navigate to:
*   Interactive Web Dashboard: `http://localhost:8000/`
*   Interactive API Swagger Documentation: `http://localhost:8000/docs`

---

## 🐳 Production Deployment

### Docker Containerization (Local or Cloud VM)
Build the production Docker container:
```bash
docker build -t vol-surface-engine .
```
Run the container locally:
```bash
docker run -p 7860:7860 vol-surface-engine
```
The application will be accessible at `http://localhost:7860`.

### Cloud Deployment (Hugging Face Spaces)
The repository is optimized for deployment on Hugging Face Spaces using the Docker SDK:
1.  Set the Space SDK to **Docker** in your Hugging Face Space settings.
2.  Ensure your `Dockerfile` exposes port `7860` (as required by Hugging Face's ingress router).
3.  Push changes to your Hugging Face Space repository:
```bash
git remote add hf https://huggingface.co/spaces/Jenak26/vol-surface-engine
git push hf master
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
