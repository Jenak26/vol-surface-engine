"""
Monte Carlo path simulation benchmark — NumPy vs Numba parallel JIT.

Run:
    python -m benchmarks.bench_mc
"""
import time
import numpy as np
from src.monte_carlo import simulate_gbm_paths, simulate_gbm_paths_fast, _NUMBA_AVAILABLE

S, T, r, sigma = 100.0, 1.0, 0.05, 0.20
N_STEPS = 252


def _bench(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000  # ms


def main():
    print("=" * 68)
    print("Monte Carlo GBM Path Simulation — NumPy vs Numba")
    if not _NUMBA_AVAILABLE:
        print("  [numba not installed — fast path falls back to numpy]")
    print("=" * 68)
    print(f"{'n_sims':>10}  {'numpy (ms)':>12}  {'numba (ms)':>12}  {'speedup':>10}")
    print("-" * 68)

    # Warm up numba JIT (first call triggers compilation, ~1 s)
    if _NUMBA_AVAILABLE:
        print("  [warming up numba JIT compilation…]", end="\r", flush=True)
        simulate_gbm_paths_fast(S, T, r, sigma, n_sims=200, n_steps=10, seed=0)
        print(" " * 45, end="\r")

    for n_sims in (10_000, 50_000, 100_000, 500_000):
        t_np = _bench(simulate_gbm_paths, S, T, r, sigma,
                      n_sims=n_sims, n_steps=N_STEPS, seed=42)
        t_nb = _bench(simulate_gbm_paths_fast, S, T, r, sigma,
                      n_sims=n_sims, n_steps=N_STEPS, seed=42)
        speedup = t_np / t_nb if t_nb > 0 else float("inf")
        print(f"{n_sims:>10,}  {t_np:>12.1f}  {t_nb:>12.1f}  {speedup:>10.1f}x")

    print("=" * 68)
    print("Note: 500k × 252 steps = 126 M path-steps per call.")
    print("Speedup depends on CPU core count and numba cache state.")


if __name__ == "__main__":
    main()
