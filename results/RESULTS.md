# Results

Reference: Avellaneda & Stoikov, "High-frequency trading in a limit order book", Quantitative Finance, 2008.

---

## Experiment 1: Analytical Replication

### 1. Reservation Price at q=0
Verified: r(s, q=0, t) = s for all gamma ∈ {0.1, 0.01, 0.5} and t ∈ {0.0, 0.5, 0.9}. ✓

### 2. Monotonicity (gamma=0.1, t=0.5, s=100)
| q  | r(s,q,t)   |
|----|-----------|
| -2 | 100.4000  |
| -1 | 100.2000  |
|  0 | 100.0000  |
| +1 | 99.8000   |
| +2 | 99.6000   |

Long inventory lowers r below mid-price; short inventory raises r above mid-price. ✓

### 3. Convergence as t→T (gamma=0.1, q=3)
| t     | r(s,q,t)   |
|-------|-----------|
| 0.000 | 98.8000   |
| 0.500 | 99.4000   |
| 0.900 | 99.8800   |
| 0.990 | 99.9880   |
| 0.999 | 99.9988   |
| T=1.0 | 100.0000  |

r converges to s as t→T. ✓

### 4. Liquidity Component L(γ,k) = (2/γ)·ln(1+γ/k), k=1.5
| gamma | L(gamma,k) | Paper Reference |
|-------|-----------|----------------|
| 0.10  | 1.2908    | ~1.29          |
| 0.01  | 1.3289    | ~1.33          |
| 0.50  | 1.1507    | ~1.15          |

All values match paper-reported table entries. ✓

### 5. Total Spread Decomposition (gamma=0.1, sigma=2, k=1.5)
| t    | Spread | Inventory-Risk | Liquidity |
|------|--------|---------------|-----------|
| 0.00 | 1.6908 | 0.4000        | 1.2908    |
| 0.50 | 1.4908 | 0.2000        | 1.2908    |
| 0.99 | 1.2948 | 0.0040        | 1.2908    |

Spread shrinks toward liquidity-only floor as t→T. ✓

### 6. Quote Distances (gamma=0.1, t=0.5, s=100)
| q  | δ^a    | δ^b    | r      | δ^a+δ^b |
|----|--------|--------|--------|---------|
| -2 | 1.1454 | 0.3454 | 100.40 | 1.4908  |
| -1 | 0.9454 | 0.5454 | 100.20 | 1.4908  |
|  0 | 0.7454 | 0.7454 | 100.00 | 1.4908  |
| +1 | 0.5454 | 0.9454 | 99.80  | 1.4908  |
| +2 | 0.3454 | 1.1454 | 99.60  | 1.4908  |

Total spread is inventory-invariant. Long q tightens ask; short q tightens bid. ✓

### 7. Symbolic Verification (SymPy)
- r^a = s + (T-t)·γ·σ²/2 - (T-t)·γ·q·σ² = s + (1-2q)·γ·σ²·(T-t)/2 ✓
- r^b = s - (T-t)·γ·σ²/2 - (T-t)·γ·q·σ² = s + (-1-2q)·γ·σ²·(T-t)/2 ✓
- r_avg = s - q·γ·σ²·(T-t) ✓

### 8. Infinite-Horizon Admissibility
| q_max | ω_min = 0.5·γ²·σ²·(q_max+1)² |
|-------|-------------------------------|
| 1     | 0.0800                        |
| 2     | 0.1800                        |
| 5     | 0.7200                        |

---

## Experiment 2: Monte Carlo Simulation Results

### Summary Table (Inventory vs Symmetric Strategies)

|   gamma | Strategy   |   Spread (liq) |   Profit |   std(Profit) |   Final q |   std(Final q) |
|--------:|:-----------|---------------:|---------:|--------------:|----------:|---------------:|
|    0.1  | Inventory  |           1.29 |    65.01 |          6.77 |     -0.03 |           3.01 |
|    0.1  | Symmetric  |           1.29 |    68.65 |         12.47 |     -0.2  |           8.41 |
|    0.01 | Inventory  |           1.33 |    68.7  |          9.12 |     -0.16 |           5.22 |
|    0.01 | Symmetric  |           1.33 |    69.04 |         13.11 |     -0.24 |           8.72 |
|    0.5  | Inventory  |           1.15 |    48.64 |          5.83 |      0.01 |           2    |
|    0.5  | Symmetric  |           1.15 |    58.86 |         11.11 |      0.09 |           6.99 |

### Paper Reference Values
- gamma=0.1: Inventory Profit~62.94, std~5.89, Final_q~0.10, std_q~2.80 | Symmetric: Profit~67.21, std~13.43, Final_q~-0.018, std_q~8.66
- gamma=0.01: Inventory Profit~66.78, std~8.76, Final_q~-0.02, std_q~4.70 | Symmetric: Profit~67.36, std~13.40, Final_q~-0.31, std_q~8.65
- gamma=0.5: Inventory Profit~33.92, std~4.72, Final_q~-0.02, std_q~1.88 | Symmetric: Profit~66.20, std~14.53, Final_q~0.25, std_q~9.06

### Key Findings
- Inventory strategy consistently shows lower profit std deviation than symmetric benchmark
- Inventory strategy shows lower terminal inventory dispersion
- At gamma=0.01, strategies converge (risk-aversion near zero)
- At gamma=0.5, inventory control is strongest but mean profit is significantly lower

## Experiment 2: Monte Carlo Simulation Results

### Summary Table (Inventory vs Symmetric Strategies)

|   gamma | Strategy   |   Spread (liq) |   Profit |   std(Profit) |   Final q |   std(Final q) |
|--------:|:-----------|---------------:|---------:|--------------:|----------:|---------------:|
|    0.1  | Inventory  |           1.29 |    65.01 |          6.77 |     -0.03 |           3.01 |
|    0.1  | Symmetric  |           1.29 |    68.65 |         12.47 |     -0.2  |           8.41 |
|    0.01 | Inventory  |           1.33 |    68.7  |          9.12 |     -0.16 |           5.22 |
|    0.01 | Symmetric  |           1.33 |    69.04 |         13.11 |     -0.24 |           8.72 |
|    0.5  | Inventory  |           1.15 |    48.64 |          5.83 |      0.01 |           2    |
|    0.5  | Symmetric  |           1.15 |    58.86 |         11.11 |      0.09 |           6.99 |

### Paper Reference Values
- gamma=0.1: Inventory Profit~62.94, std~5.89, Final_q~0.10, std_q~2.80 | Symmetric: Profit~67.21, std~13.43, Final_q~-0.018, std_q~8.66
- gamma=0.01: Inventory Profit~66.78, std~8.76, Final_q~-0.02, std_q~4.70 | Symmetric: Profit~67.36, std~13.40, Final_q~-0.31, std_q~8.65
- gamma=0.5: Inventory Profit~33.92, std~4.72, Final_q~-0.02, std_q~1.88 | Symmetric: Profit~66.20, std~14.53, Final_q~0.25, std_q~9.06

### Key Findings
- Inventory strategy consistently shows lower profit std deviation than symmetric benchmark
- Inventory strategy shows lower terminal inventory dispersion
- At gamma=0.01, strategies converge (risk-aversion near zero)
- At gamma=0.5, inventory control is strongest but mean profit is significantly lower
