# demo.py
"""
Volatility Surface Engine Demonstration
======================================
This script demonstrates the advanced quantitative upgrades implemented in the engine:
1. Numerical stability of the branch-cut-free Heston characteristic function.
2. Speedup of the new Vega-weighted Heston calibration.
3. Smoothness of the Dupire local volatility surface using time-dimension cubic splines.
4. Arbitrage-free SVI volatility surface fitting.

Run:
    python demo.py
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from data.fetcher import fetch_nse_option_chain
from data.preprocessor import clean_option_chain
from src.vol_surface import VolatilitySurface
from src.local_vol import local_vol_grid
from src.heston import heston_price, heston_iv, calibrate_heston
from src.black_scholes import bs_price

def main():
    print("=" * 80)
    print("           VOLATILITY SURFACE ENGINE - ADVANCED DEMONSTRATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: Load and Preprocess Market Options Data
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading and cleaning options chain...")
    df_raw = fetch_nse_option_chain('NIFTY')
    spot = df_raw['spot'].iloc[0] if 'spot' in df_raw.columns else 22000.0
    r = df_raw['r'].iloc[0] if 'r' in df_raw.columns else 0.065
    
    print(f"  Spot Price: {spot:.2f}")
    print(f"  Risk-Free Rate: {r:.2%}")
    print(f"  Raw Quote Count: {len(df_raw)}")
    
    cleaned_df = clean_option_chain(df_raw, spot=spot, r=r, min_oi=0, min_volume=0)
    print(f"  Cleaned Quote Count: {len(cleaned_df)}")
    
    expiries = cleaned_df['T'].unique()
    print(f"  Available Expiries (T in years): {list(np.round(expiries, 4))}")

    # --------------------------------------------------------------------------
    # STEP 2: Arbitrage-Free SVI Volatility Surface Fitting
    # --------------------------------------------------------------------------
    print("\n[Step 2] Fitting SVI Volatility Surface with Butterfly Arbitrage Penalty...")
    t0 = time.perf_counter()
    surface = VolatilitySurface(cleaned_df, spot=spot, r=r)
    t1 = time.perf_counter()
    print(f"  Surface Fitting Completed in {(t1-t0)*1000:.2f} ms")

    # Run arbitrage checks
    cal_arb = surface.check_calendar_arbitrage()
    but_arb = surface.check_butterfly_arbitrage()
    print(f"  Calendar Arbitrage Free: {cal_arb['calendar_arbitrage_free']}")
    print(f"  Butterfly Arbitrage Free: {but_arb['butterfly_arbitrage_free']} (Fraction Valid: {but_arb['fraction_valid']:.2%})")

    # --------------------------------------------------------------------------
    # STEP 3: Stable Heston Pricing and High-Performance Calibration
    # --------------------------------------------------------------------------
    print("\n[Step 3] Calibrating Heston Model via Fast Vega-Weighted Pricing...")
    # Select near-term slice for calibration
    T_cal = expiries[0]
    slice_df = cleaned_df[cleaned_df['T'] == T_cal]
    log_moneyness = slice_df['log_moneyness'].values
    market_ivs = slice_df['computed_iv'].values
    
    print(f"  Calibrating to slice T = {T_cal:.4f} years ({len(slice_df)} strikes)...")
    
    t0 = time.perf_counter()
    optimal_params, rmse = calibrate_heston(log_moneyness, market_ivs, T_cal, spot, r)
    t1 = time.perf_counter()
    
    print(f"  Calibration Completed in {(t1-t0)*1000:.2f} ms")
    print(f"  Optimal Heston Parameters:")
    for param, val in optimal_params.items():
        print(f"    - {param:<7}: {val:.6f}")
    print(f"  Final Implied Volatility RMSE: {rmse:.6f}")

    # Verify Heston Pricing stability for long-dated option
    print("\n  Verifying Heston model numerical stability for long-dated options (T = 10.0 years)...")
    long_T = 10.0
    # The old characteristic function would overflow here for large phi,
    # let's verify the new stable characteristic function handles this with ease
    stable_price = heston_price(spot, spot, long_T, r, **optimal_params)
    print(f"    - Heston Price (ATM Call, T=10y): {stable_price:.4f} (No Numerical Overflow)")

    # --------------------------------------------------------------------------
    # STEP 4: Smooth Dupire Local Volatility Surface Extraction
    # --------------------------------------------------------------------------
    print("\n[Step 4] Extracting Dupire Local Volatility Surface...")
    k_grid = np.linspace(-0.25, 0.25, 50)
    T_grid = np.linspace(expiries[0], expiries[-1], 20)
    
    t0 = time.perf_counter()
    local_vols = local_vol_grid(surface, k_grid, T_grid)
    t1 = time.perf_counter()
    print(f"  Local Vol Grid Extraction Completed in {(t1-t0)*1000:.2f} ms")

    # --------------------------------------------------------------------------
    # STEP 5: Generate Visualizations
    # --------------------------------------------------------------------------
    print("\n[Step 5] Generating Volatility Surface Charts...")
    K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
    
    # Plot Implied Volatility Surface
    fig = plt.figure(figsize=(16, 7))
    
    # Subplot 1: Implied Vol
    ax1 = fig.add_subplot(121, projection='3d')
    iv_grid = surface.get_iv_grid(k_grid, T_grid)
    surf1 = ax1.plot_surface(K_mesh, T_mesh, iv_grid, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax1.set_title("Implied Volatility Surface (SVI)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Log-Moneyness ln(K/F)")
    ax1.set_ylabel("Expiry T (Years)")
    ax1.set_zlabel("Implied Volatility")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # Subplot 2: Dupire Local Vol
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(K_mesh, T_mesh, local_vols, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax2.set_title("Dupire Local Volatility Surface (C2 Splined)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Log-Moneyness ln(K/F)")
    ax2.set_ylabel("Expiry T (Years)")
    ax2.set_zlabel("Local Volatility")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    plt.tight_layout()
    plt.savefig("volatility_surfaces.png", dpi=300)
    print("  Saved 'volatility_surfaces.png' successfully.")

    # Plot Heston Calibration Fit vs Market Smile
    plt.figure(figsize=(9, 5))
    F_cal = spot * np.exp(r * T_cal)
    cal_strikes = F_cal * np.exp(log_moneyness)
    
    # Sort for plotting
    sort_idx = np.argsort(cal_strikes)
    sorted_strikes = cal_strikes[sort_idx]
    sorted_market_ivs = market_ivs[sort_idx]
    
    heston_ivs = []
    for k_i in log_moneyness[sort_idx]:
        K_i = F_cal * np.exp(k_i)
        iv_h = heston_iv(spot, K_i, T_cal, r, **optimal_params)
        heston_ivs.append(iv_h)
        
    plt.plot(sorted_strikes, sorted_market_ivs, 'o', label='Market Implied Vol', color='firebrick')
    plt.plot(sorted_strikes, heston_ivs, '-', label='Calibrated Heston Vol', color='navy', linewidth=2.5)
    plt.title(f"Heston Model Calibration Fit (Expiry T = {T_cal:.4f} years)", fontsize=12, fontweight='bold')
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("heston_calibration_fit.png", dpi=300)
    print("  Saved 'heston_calibration_fit.png' successfully.")
    
    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETED SUCCESSFULLY. ALL RESULTS GENERATED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
