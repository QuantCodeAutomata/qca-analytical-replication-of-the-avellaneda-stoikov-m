# Avellaneda-Stoikov Market-Making Model

A Python replication of the Avellaneda & Stoikov (2008) high-frequency market-making model, covering both the analytical closed-form solution and Monte Carlo simulation experiments.

---

## Background

Avellaneda & Stoikov (2008) model the problem of a market maker who continuously posts limit orders on both sides of the book. The agent maximises expected utility of terminal wealth while managing inventory risk. The key outputs are:

- **Reservation price** — the mid-price adjusted for inventory and risk aversion.
- **Optimal spread** — the symmetric or asymmetric spread placed around the reservation price.
- **Value function** — the solution to the Hamilton-Jacobi-Bellman (HJB) equation under a Gaussian diffusion for the mid-price.

The model admits an exact closed-form solution in the finite-horizon case and a simple infinite-horizon approximation, both of which are implemented here.

---

## Repository Structure

```
project/
├── README.md
├── .gitignore
├── data/                          # Raw / synthetic price data (git-ignored binaries)
│   └── .gitkeep
├── results/                       # Generated plots and summary tables
│   └── RESULTS.md                 # Human-readable experiment results
├── src/
│   ├── __init__.py
│   └── avellaneda_stoikov/
│       ├── __init__.py            # Public API re-exports
│       ├── formulas.py            # Closed-form analytical expressions
│       ├── simulator.py           # Mid-price path simulation & order-fill logic
│       └── experiments.py        # Orchestration of Experiment 1 & 2
├── exp/
│   ├── __init__.py
│   ├── run_exp1.py                # CLI entry point – Experiment 1
│   └── run_exp2.py                # CLI entry point – Experiment 2
└── tests/
    └── __init__.py
```

---

## Model Equations

### Reservation Price

```
r(s, t) = s - q · γ · σ² · (T - t)
```

where `s` is the current mid-price, `q` is inventory, `γ` is the risk-aversion coefficient, `σ` is mid-price volatility, and `T - t` is time to horizon.

### Optimal Spread

```
δ_ask + δ_bid = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
```

where `κ` is the order-book depth parameter governing the arrival-rate decay of market orders.

### Infinite-Horizon Approximation

As `T → ∞` the spread simplifies to:

```
δ_ask + δ_bid ≈ γ · σ² / (2·λ) + sqrt(σ²·γ / (2·λ)) · (1 + γ/κ)^(1 + κ/γ)
```

---

## Experiments

### Experiment 1 – Analytical Replication

Verifies the closed-form formulas against the theoretical predictions from the paper:

- Reservation price as a function of inventory `q` and time `t`.
- Bid/ask distances and total spread as a function of time-to-maturity.
- Admissibility conditions for the value function.
- Frozen value-function surface over `(q, t)`.

Produces plots saved to `results/`.

### Experiment 2 – Monte Carlo Simulation

Runs a large number of GBM / arithmetic-Brownian-motion mid-price paths and simulates two competing strategies:

| Strategy | Description |
|---|---|
| **AS Inventory** | Posts quotes at the AS reservation price ± optimal half-spread. |
| **Symmetric benchmark** | Posts quotes at mid ± fixed symmetric half-spread. |

Reports terminal P&L distributions, Sharpe ratios, maximum inventory excursion, and fill-rate statistics. Results are written to `results/RESULTS.md` and visualised in `results/`.

---

## Usage

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Run Experiment 1

```bash
python exp/run_exp1.py
```

### Run Experiment 2

```bash
python exp/run_exp2.py
```

### Run Both via the Public API

```python
from src.avellaneda_stoikov import run_all_experiments
run_all_experiments()
```

### Use Individual Formulas

```python
from src.avellaneda_stoikov import (
    reservation_price,
    total_spread,
    infinite_horizon_reservation_ask,
    infinite_horizon_reservation_bid,
)

r = reservation_price(s=100.0, q=2, gamma=0.1, sigma=2.0, T=1.0, t=0.5)
spread = total_spread(gamma=0.1, sigma=2.0, T=1.0, t=0.5, kappa=1.5)
```

---

## Parameters

| Symbol | Argument | Typical value | Description |
|--------|----------|---------------|-------------|
| `σ` | `sigma` | `2.0` | Mid-price volatility ($/s^½) |
| `γ` | `gamma` | `0.1` | Absolute risk-aversion coefficient |
| `κ` | `kappa` | `1.5` | Order-book depth / fill-rate decay |
| `T` | `T` | `1.0` | Trading horizon (normalised) |
| `λ` | `lam` | `140` | Baseline market-order arrival rate |
| `A` | `A` | `140` | Arrival-rate pre-factor (= `λ`) |

---

## Reference

> Avellaneda, M., & Stoikov, S. (2008). **High-frequency trading in a limit order book**. *Quantitative Finance*, 8(3), 217–224. https://doi.org/10.1080/14697680701381228
