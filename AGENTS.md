# Project: Avellaneda-Stoikov Market-Making Model

## Repository Layout

```
src/
  __init__.py
  avellaneda_stoikov/
    __init__.py      # re-exports all public symbols; imports simulator & experiments too
    formulas.py      # DONE — analytical closed-form expressions
    simulator.py     # DONE — Monte Carlo simulator (SimConfig, simulate_path, run_experiment)
    experiments.py   # DONE — orchestration of Exp 1 & 2
exp/
  run_exp1.py        # CLI entry for Experiment 1
  run_exp2.py        # CLI entry for Experiment 2
tests/               # test suite (no files yet)
```

## Important: Package Import Caveat

`src/avellaneda_stoikov/__init__.py` may import from `simulator` and `experiments`.
Use `sys.path.insert(0, '/workspace/project')` (already done in `exp/run_expN.py`)
and import directly: `from src.avellaneda_stoikov.experiments import ...`

## Formulas Module (`formulas.py`) — Implemented

### Finite-horizon functions (all depend on tau = T - t)

| Function | Formula |
|---|---|
| `frozen_value_function` | `-exp(-γx) · exp(-γqs) · exp(γ²q²σ²τ/2)` |
| `reservation_ask_frozen` | `s + (1-2q)·γ·σ²·τ/2` |
| `reservation_bid_frozen` | `s + (-1-2q)·γ·σ²·τ/2` |
| `reservation_price` | `s - q·γ·σ²·τ`  (mean of r^a and r^b) |
| `liquidity_component` | `(2/γ)·ln(1 + γ/k)` |
| `total_spread` | `γ·σ²·τ + (2/γ)·ln(1 + γ/k)` |
| `ask_distance` | `(1-2q)·γ·σ²·τ/2 + (1/γ)·ln(1+γ/k)` |
| `bid_distance` | `(1+2q)·γ·σ²·τ/2 + (1/γ)·ln(1+γ/k)` |
| `symmetric_half_spread` | `γ·σ²·τ/2 + (1/γ)·ln(1+γ/k)` (q=0 case) |

### Key identities (verified numerically and symbolically)

- `(r^a + r^b) / 2 == reservation_price` for all q
- `ask_distance(q=0) == bid_distance(q=0) == symmetric_half_spread`
- `ask_distance + bid_distance == total_spread` for all q (inventory-invariant!)
- `2 * symmetric_half_spread == total_spread`

### Infinite-horizon functions (simplified linear approximation)

| Function | Formula |
|---|---|
| `infinite_horizon_reservation_ask` | `s - q·γ·σ²/ω + γ·σ²/(2ω)` |
| `infinite_horizon_reservation_bid` | `s - q·γ·σ²/ω - γ·σ²/(2ω)` |
| `admissibility_condition` | `0.5·γ²·σ²·q²` (minimum ω) |

- Infinite-horizon spread: `rbar^a - rbar^b = γ·σ²/ω`

### Symbolic verification

`verify_frozen_value_symbolic()` uses SymPy to derive r^a and r^b from the
indifference equations and confirms they equal the paper's closed-form expressions.
SymPy 1.14 is available in the environment.

## Simulator Module (`simulator.py`) — Implemented

### Key classes

| Class | Purpose |
|---|---|
| `SimConfig` | Dataclass holding all simulation parameters (paper defaults: s0=100, T=1, dt=0.005, sigma=2, A=140, k=1.5, gamma=0.1, n_paths=1000, seed=42) |
| `PathResult` | Per-path output: final_profit, final_inventory, optional trajectory arrays |
| `ExperimentResult` | Aggregated MC output: mean/std of profit and final_q, plus raw arrays |

### Key functions

- `simulate_path(config, strategy, price_moves, ask_uniforms, bid_uniforms, store_trajectory=False)` — one MC path
- `run_experiment(config)` → `(inv_result, sym_result)` — 1000-path CRN experiment

### Critical implementation note: probability clamping

The Bernoulli approximation `p_fill = lambda * dt` can exceed 1.0 when inventory
is extreme (|q| ≥ 3 with default params), causing `delta < 0` (quote crosses mid).
`simulate_path` clamps `p_fill = min(lambda * dt, 1.0)` — physically correct
(quote at/above mid means certain fill). **Do not restore hard assertions.**

### Strategies

- `'inventory'`: delta_a = (1-2q)*half_inv_risk + c_gamma, delta_b = (1+2q)*half_inv_risk + c_gamma
- `'symmetric'`: delta_a = delta_b = half_spread = 0.5*gamma*sigma^2*(T-t) + c_gamma

where `c_gamma = (1/gamma)*ln(1+gamma/k)` (liquidity half-spread)
and `half_inv_risk = 0.5*gamma*sigma^2*(T-t)`

### Exp 2 results (seed=42, 1000 paths)

| gamma | Strategy | Spread (liq) | Profit | std(Profit) | Final q | std(Final q) |
|---|---|---|---|---|---|---|
| 0.10 | Inventory | 1.29 | 65.01 | 6.77 | -0.03 | 3.01 |
| 0.10 | Symmetric | 1.29 | 68.65 | 12.47 | -0.20 | 8.41 |
| 0.01 | Inventory | 1.33 | 68.70 | 9.12 | -0.16 | 5.22 |
| 0.01 | Symmetric | 1.33 | 69.04 | 13.11 | -0.24 | 8.72 |
| 0.50 | Inventory | 1.15 | 48.64 | 5.83 | 0.01 | 2.00 |
| 0.50 | Symmetric | 1.15 | 58.86 | 11.11 | 0.09 | 6.99 |

Key finding: inventory strategy consistently lower std(Profit) and std(Final q) vs symmetric.

## Dependencies

- `numpy` — all numerical functions
- `sympy` — `verify_frozen_value_symbolic()` only (imported at module top level)
- `pandas`, `matplotlib` — used in `experiments.py` for tables and plots
- Python 3.x with `typing.Union` (available)

## Parameter Conventions

- `gamma` — CARA risk-aversion coefficient (> 0)
- `sigma` — mid-price ABM volatility (> 0)
- `k` — exponential decay rate of Poisson order-arrival intensity (> 0)
- `omega` — discount rate for infinite-horizon problem (> 0)
- `tau = T - t` — time remaining (computed inside each function)
