"""
Avellaneda-Stoikov Market-Making Model: Monte Carlo Simulator

Reference: Avellaneda & Stoikov, 'High-frequency trading in a limit order book',
Quantitative Finance, 2008, Section 3.3.

Simulates the finite-horizon inventory-based quoting strategy vs symmetric benchmark
using the paper's exact parameter set and discrete-time approximation.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimConfig:
    """Simulation configuration matching paper parameters exactly."""
    s0: float = 100.0          # initial mid-price
    T: float = 1.0             # terminal horizon
    dt: float = 0.005          # time step
    sigma: float = 2.0         # mid-price volatility
    A: float = 140.0           # execution intensity scale
    k: float = 1.5             # intensity decay
    gamma: float = 0.1         # risk aversion
    n_paths: int = 1000        # number of Monte Carlo paths
    seed: int = 42             # random seed
    common_random_numbers: bool = True  # use CRN for fair comparison
    q0: float = 0.0            # initial inventory
    X0: float = 0.0            # initial cash


@dataclass
class PathResult:
    """Results from a single simulated path."""
    final_profit: float
    final_inventory: float
    # Optional: store full trajectory for plotting
    mid_prices: Optional[np.ndarray] = None
    reservation_prices: Optional[np.ndarray] = None
    ask_quotes: Optional[np.ndarray] = None
    bid_quotes: Optional[np.ndarray] = None


@dataclass
class ExperimentResult:
    """Aggregated results from Monte Carlo experiment."""
    strategy: str
    gamma: float
    spread: float  # liquidity component
    mean_profit: float
    std_profit: float
    mean_final_q: float
    std_final_q: float
    profits: np.ndarray = field(default_factory=lambda: np.array([]))
    final_qs: np.ndarray = field(default_factory=lambda: np.array([]))


def simulate_path(
    config: SimConfig,
    strategy: str,              # 'inventory' or 'symmetric'
    price_moves: np.ndarray,    # pre-generated price direction array (+1/-1), shape (N,)
    ask_uniforms: np.ndarray,   # pre-generated uniform draws for ask fills, shape (N,)
    bid_uniforms: np.ndarray,   # pre-generated uniform draws for bid fills, shape (N,)
    store_trajectory: bool = False
) -> PathResult:
    """
    Simulate one Monte Carlo path for either inventory or symmetric strategy.

    Uses the paper's binomial ABM approximation for mid-price:
        S_{t+dt} = S_t + sigma*sqrt(dt) with prob 0.5
        S_{t+dt} = S_t - sigma*sqrt(dt) with prob 0.5

    Fill events use Bernoulli approximation to Poisson:
        P(ask fill in [t, t+dt]) = lambda^a(delta^a) * dt
        P(bid fill in [t, t+dt]) = lambda^b(delta^b) * dt

    Args:
        config: SimConfig with all parameters
        strategy: 'inventory' uses reservation price centering; 'symmetric' centers at mid
        price_moves: pre-generated array of +1/-1 for binomial mid-price moves
        ask_uniforms: uniform [0,1] draws for ask fill Bernoulli trials
        bid_uniforms: uniform [0,1] draws for bid fill Bernoulli trials
        store_trajectory: if True, store full path for plotting
    Returns:
        PathResult with final P&L and inventory
    """
    N = int(config.T / config.dt)
    assert len(price_moves) == N, f'Expected {N} price moves'

    s = config.s0
    q = config.q0
    X = config.X0

    # Precompute constants
    c_gamma = (1.0 / config.gamma) * np.log(1.0 + config.gamma / config.k)  # liquidity component half
    sqrt_dt = np.sqrt(config.dt)

    if store_trajectory:
        mid_prices = np.zeros(N + 1)
        res_prices = np.zeros(N + 1)
        ask_quotes = np.zeros(N + 1)
        bid_quotes = np.zeros(N + 1)
        mid_prices[0] = s

    for n in range(N):
        t = n * config.dt
        remaining = config.T - t

        # Compute quote distances based on strategy
        if strategy == 'inventory':
            # Inventory strategy: center at reservation price
            # delta^a = ((1 - 2q) * gamma * sigma^2 * (T-t)) / 2 + c_gamma
            # delta^b = ((1 + 2q) * gamma * sigma^2 * (T-t)) / 2 + c_gamma
            half_inv_risk = 0.5 * config.gamma * config.sigma**2 * remaining
            delta_a = (1 - 2 * q) * half_inv_risk + c_gamma
            delta_b = (1 + 2 * q) * half_inv_risk + c_gamma
        else:  # symmetric
            # Symmetric: same spread but centered at mid-price
            half_spread = 0.5 * config.gamma * config.sigma**2 * remaining + c_gamma
            delta_a = half_spread
            delta_b = half_spread

        # Quote prices
        p_a = s + delta_a
        p_b = s - delta_b

        if store_trajectory:
            r = s - q * config.gamma * config.sigma**2 * remaining
            res_prices[n] = r
            ask_quotes[n] = p_a
            bid_quotes[n] = p_b

        # Fill intensities and Bernoulli probabilities
        lambda_a = config.A * np.exp(-config.k * delta_a)
        lambda_b = config.A * np.exp(-config.k * delta_b)
        # Clamp to [0, 1]: delta < 0 (quote crosses mid) means certain fill;
        # the Bernoulli approximation is only valid for small lambda*dt.
        p_fill_a = min(lambda_a * config.dt, 1.0)
        p_fill_b = min(lambda_b * config.dt, 1.0)

        # Bernoulli fill indicators using pre-generated uniforms
        fill_a = ask_uniforms[n] < p_fill_a  # ask fill (dealer sells)
        fill_b = bid_uniforms[n] < p_fill_b  # bid fill (dealer buys)

        # Update state: fills first, then mid-price
        if fill_a:
            q -= 1
            X += p_a
        if fill_b:
            q += 1
            X -= p_b

        # Update mid-price: binomial ABM
        s += price_moves[n] * config.sigma * sqrt_dt

        if store_trajectory:
            mid_prices[n + 1] = s

    if store_trajectory:
        res_prices[N] = s - q * config.gamma * config.sigma**2 * 0.0  # = s at T
        ask_quotes[N] = s
        bid_quotes[N] = s
        return PathResult(
            final_profit=X + q * s,
            final_inventory=q,
            mid_prices=mid_prices,
            reservation_prices=res_prices,
            ask_quotes=ask_quotes,
            bid_quotes=bid_quotes
        )

    return PathResult(final_profit=X + q * s, final_inventory=q)


def run_experiment(
    config: SimConfig,
) -> tuple[ExperimentResult, ExperimentResult]:
    """
    Run 1000-path Monte Carlo for both inventory and symmetric strategies.
    Uses Common Random Numbers for fair comparison.

    Returns:
        Tuple of (inventory_result, symmetric_result)
    """
    rng = np.random.default_rng(config.seed)
    N = int(config.T / config.dt)

    profits_inv = np.zeros(config.n_paths)
    profits_sym = np.zeros(config.n_paths)
    qs_inv = np.zeros(config.n_paths)
    qs_sym = np.zeros(config.n_paths)

    # Liquidity component (for spread column in table)
    liq = (2.0 / config.gamma) * np.log(1.0 + config.gamma / config.k)

    for i in range(config.n_paths):
        # Generate shared random arrays (CRN)
        price_moves = rng.choice([-1, 1], size=N).astype(float)
        ask_uniforms = rng.uniform(0, 1, size=N)
        bid_uniforms = rng.uniform(0, 1, size=N)

        # Inventory strategy
        res_inv = simulate_path(config, 'inventory', price_moves, ask_uniforms, bid_uniforms)
        profits_inv[i] = res_inv.final_profit
        qs_inv[i] = res_inv.final_inventory

        # Symmetric strategy (same random numbers)
        res_sym = simulate_path(config, 'symmetric', price_moves, ask_uniforms, bid_uniforms)
        profits_sym[i] = res_sym.final_profit
        qs_sym[i] = res_sym.final_inventory

    inv_result = ExperimentResult(
        strategy='Inventory',
        gamma=config.gamma,
        spread=round(liq, 2),
        mean_profit=round(float(np.mean(profits_inv)), 2),
        std_profit=round(float(np.std(profits_inv, ddof=1)), 2),
        mean_final_q=round(float(np.mean(qs_inv)), 2),
        std_final_q=round(float(np.std(qs_inv, ddof=1)), 2),
        profits=profits_inv,
        final_qs=qs_inv,
    )
    sym_result = ExperimentResult(
        strategy='Symmetric',
        gamma=config.gamma,
        spread=round(liq, 2),
        mean_profit=round(float(np.mean(profits_sym)), 2),
        std_profit=round(float(np.std(profits_sym, ddof=1)), 2),
        mean_final_q=round(float(np.mean(qs_sym)), 2),
        std_final_q=round(float(np.std(qs_sym, ddof=1)), 2),
        profits=profits_sym,
        final_qs=qs_sym,
    )
    return inv_result, sym_result
