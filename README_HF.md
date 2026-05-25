---
title: Vol Surface Engine
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Volatility Surface Engine

A production-grade derivatives pricing and calibration engine built from first principles — from Black-Scholes to a calibrated Heston stochastic volatility model with Dupire local volatility extraction.

---

## 🚀 How to Use the Interactive API Console

This Hugging Face Space hosts a **live, fully interactive API console** where you can execute quantitative pricing calls directly from your browser.

1.  **Open the Swagger Documentation Page:**
    👉 **[Click Here to Open /docs](https://jenak26-vol-surface-engine.hf.space/docs)**
2.  **Try it out:**
    *   Expand the **`POST /price`** endpoint.
    *   Click **"Try it out"** in the top right of the endpoint box.
    *   Paste one of the sample payloads below into the text area.
    *   Click the blue **"Execute"** button and scroll down to see the results.

### Sample Payload 1: Black-Scholes ATM Call Option
```json
{
  "S": 100,
  "K": 100,
  "T": 1.0,
  "r": 0.05,
  "sigma": 0.20,
  "model": "bs",
  "option_type": "call"
}
```

### Sample Payload 2: Calibrated Heston Stochastic Volatility Call
```json
{
  "S": 100,
  "K": 100,
  "T": 1.0,
  "r": 0.05,
  "model": "heston",
  "option_type": "call",
  "kappa": 1.5,
  "theta": 0.04,
  "sigma_v": 0.3,
  "rho_v": -0.5,
  "v0": 0.04
}
```

### Sample Payload 3: American Put via Longstaff-Schwartz Monte Carlo (LSM)
```json
{
  "S": 100,
  "K": 100,
  "T": 1.0,
  "r": 0.05,
  "sigma": 0.20,
  "model": "lsm",
  "option_type": "put",
  "n_sims": 50000
}
```

---

## 📈 Quantitative Engine Highlights

*   **Stable Heston Pricing:** Implements the branch-cut-free Heston characteristic function (Gatheral 2006, Albrecher 2007), eliminating exponential overflows for long maturities.
*   **Fast Vega-Weighted Calibration:** Calibrates to market options chains in milliseconds by optimizing directly on Vega-weighted price errors, removing the root-finding Newton-Raphson loop from the objective function.
*   **Arbitrage-Free SVI Surface:** Fits SVI models to expiry slices under butterfly and calendar spread no-arbitrage constraints.
*   **Smooth Dupire Local Volatility:** Applies time-dimension natural cubic spline ($C^2$ continuous) interpolation to total variance, enabling exact analytical time derivatives ($\partial w / \partial T$) and removing all step-like vertical cliffs.
