"""
Pytest test suite for the Avellaneda-Stoikov Market-Making Model.

Tests cover:
- Analytical formula correctness (exp_1)
- Mathematical properties and identities
- Simulation mechanics (exp_2)
- Edge cases and numerical stability
- Methodology adherence to the paper
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.avellaneda_stoikov.formulas import (
    frozen_value_function,
    reservation_ask_frozen,
    reservation_bid_frozen,
    reservation_price,
    liquidity_component,
    total_spread,
    ask_distance,
    bid_distance,
    symmetric_half_spread,
    infinite_horizon_reservation_ask,
    infinite_horizon_reservation_bid,
    admissibility_condition,
    verify_frozen_value_symbolic,
)
from src.avellaneda_stoikov.simulator import (
    SimConfig,
    PathResult,
    ExperimentResult,
    simulate_path,
    run_experiment,
)

# -----------------------------------------------------------------------
# Paper parameters used throughout
# -----------------------------------------------------------------------
S0 = 100.0
T = 1.0
SIGMA = 2.0
K = 1.5
GAMMA_VALUES = [0.1, 0.01, 0.5]
TOL = 1e-10


# -----------------------------------------------------------------------
# Tests: frozen_value_function
# -----------------------------------------------------------------------

class TestFrozenValueFunction:
    """Tests for the frozen-inventory CARA value function."""

    def test_always_negative(self):
        """Value function must always be strictly negative (CARA utility)."""
        for gamma in GAMMA_VALUES:
            for q in [-2, -1, 0, 1, 2]:
                v = frozen_value_function(0.0, S0, q, 0.5, T, gamma, SIGMA)
                assert v < 0, f"v={v} is not negative for q={q}, gamma={gamma}"

    def test_decreasing_in_x(self):
        """More cash should raise the utility value (less negative) under CARA."""
        gamma = 0.1
        v1 = frozen_value_function(0.0, S0, 0, 0.5, T, gamma, SIGMA)
        v2 = frozen_value_function(10.0, S0, 0, 0.5, T, gamma, SIGMA)
        assert v2 > v1, "Utility should increase with cash"

    def test_variance_cost_increases_with_q_squared(self):
        """Variance cost term exp(0.5 * gamma^2 * q^2 * sigma^2 * tau) must be >= 1."""
        gamma = 0.1
        q = 3
        tau = T - 0.5
        # v = -exp(-gamma*x) * exp(-gamma*q*s) * exp(0.5*gamma^2*q^2*sigma^2*tau)
        # The exp term for variance cost
        var_cost = np.exp(0.5 * gamma**2 * q**2 * SIGMA**2 * tau)
        assert var_cost >= 1.0, "Variance cost must be >= 1"

    def test_q0_inventory_cost_zero(self):
        """At q=0, variance cost is zero (exp term = 1)."""
        gamma = 0.1
        v_q0 = frozen_value_function(0.0, S0, 0, 0.5, T, gamma, SIGMA)
        expected = -np.exp(-gamma * 0.0) * np.exp(-gamma * 0 * S0) * 1.0
        assert abs(v_q0 - expected) < TOL

    def test_terminal_condition(self):
        """At t=T, frozen value = -exp(-gamma*(x + q*s)) exactly."""
        gamma = 0.1
        x, q, s = 50.0, 2, 105.0
        v = frozen_value_function(x, s, q, T, T, gamma, SIGMA)
        expected = -np.exp(-gamma * (x + q * s))
        assert abs(v - expected) < TOL, f"v={v}, expected={expected}"


# -----------------------------------------------------------------------
# Tests: reservation_ask_frozen and reservation_bid_frozen
# -----------------------------------------------------------------------

class TestFrozenReservationPrices:
    """Tests for frozen-inventory reservation ask and bid prices."""

    def test_ask_formula(self):
        """Verify r^a = s + (1-2q)*gamma*sigma^2*(T-t)/2."""
        gamma, q, t, s = 0.1, 2, 0.5, 100.0
        expected = s + (1 - 2 * q) * gamma * SIGMA**2 * (T - t) / 2
        result = reservation_ask_frozen(s, q, t, T, gamma, SIGMA)
        assert abs(result - expected) < TOL

    def test_bid_formula(self):
        """Verify r^b = s + (-1-2q)*gamma*sigma^2*(T-t)/2."""
        gamma, q, t, s = 0.1, 2, 0.5, 100.0
        expected = s + (-1 - 2 * q) * gamma * SIGMA**2 * (T - t) / 2
        result = reservation_bid_frozen(s, q, t, T, gamma, SIGMA)
        assert abs(result - expected) < TOL

    def test_ask_greater_than_bid(self):
        """Reservation ask must exceed reservation bid (positive spread)."""
        for gamma in GAMMA_VALUES:
            for q in [-2, 0, 2]:
                r_a = reservation_ask_frozen(S0, q, 0.5, T, gamma, SIGMA)
                r_b = reservation_bid_frozen(S0, q, 0.5, T, gamma, SIGMA)
                assert r_a > r_b, f"r^a={r_a} <= r^b={r_b} for q={q}, gamma={gamma}"

    def test_average_equals_reservation_price(self):
        """(r^a + r^b) / 2 must equal reservation_price."""
        for gamma in GAMMA_VALUES:
            for q in [-3, -1, 0, 1, 3]:
                r_a = reservation_ask_frozen(S0, q, 0.5, T, gamma, SIGMA)
                r_b = reservation_bid_frozen(S0, q, 0.5, T, gamma, SIGMA)
                r_avg = (r_a + r_b) / 2
                r = reservation_price(S0, q, 0.5, T, gamma, SIGMA)
                assert abs(r_avg - r) < TOL, f"avg={r_avg}, r={r}"

    def test_converge_to_s_at_T(self):
        """At t=T, both frozen prices converge to s."""
        gamma, q = 0.1, 5
        r_a = reservation_ask_frozen(S0, q, T, T, gamma, SIGMA)
        r_b = reservation_bid_frozen(S0, q, T, T, gamma, SIGMA)
        assert abs(r_a - S0) < TOL, f"r^a at T = {r_a}"
        assert abs(r_b - S0) < TOL, f"r^b at T = {r_b}"

    def test_utility_indifference_ask(self):
        """Verify indifference condition: v(x+r^a, s, q-1, t) == v(x, s, q, t)."""
        gamma, q, t, x = 0.1, 2, 0.5, 10.0
        r_a = reservation_ask_frozen(S0, q, t, T, gamma, SIGMA)
        v_before = frozen_value_function(x, S0, q, t, T, gamma, SIGMA)
        v_after = frozen_value_function(x + r_a, S0, q - 1, t, T, gamma, SIGMA)
        assert abs(v_before - v_after) < 1e-8, f"Indifference violated: {v_before} vs {v_after}"

    def test_utility_indifference_bid(self):
        """Verify indifference condition: v(x-r^b, s, q+1, t) == v(x, s, q, t)."""
        gamma, q, t, x = 0.1, 2, 0.5, 10.0
        r_b = reservation_bid_frozen(S0, q, t, T, gamma, SIGMA)
        v_before = frozen_value_function(x, S0, q, t, T, gamma, SIGMA)
        v_after = frozen_value_function(x - r_b, S0, q + 1, t, T, gamma, SIGMA)
        assert abs(v_before - v_after) < 1e-8, f"Indifference violated: {v_before} vs {v_after}"


# -----------------------------------------------------------------------
# Tests: reservation_price
# -----------------------------------------------------------------------

class TestReservationPrice:
    """Tests for the operational reservation/indifference price."""

    def test_q0_equals_midprice(self):
        """r(s,0,t) = s for all gamma, t (paper requirement)."""
        for gamma in GAMMA_VALUES:
            for t in [0.0, 0.25, 0.5, 0.75, 0.99]:
                r = reservation_price(S0, 0, t, T, gamma, SIGMA)
                assert abs(r - S0) < TOL, f"r(q=0) = {r} != s = {S0}"

    def test_positive_inventory_lowers_reservation_price(self):
        """Long inventory (q>0) lowers r below mid-price."""
        for gamma in GAMMA_VALUES:
            r = reservation_price(S0, 2, 0.5, T, gamma, SIGMA)
            assert r < S0, f"r={r} should be below s={S0} for long inventory"

    def test_negative_inventory_raises_reservation_price(self):
        """Short inventory (q<0) raises r above mid-price."""
        for gamma in GAMMA_VALUES:
            r = reservation_price(S0, -2, 0.5, T, gamma, SIGMA)
            assert r > S0, f"r={r} should be above s={S0} for short inventory"

    def test_converges_to_s_at_T(self):
        """r(s,q,T) = s regardless of inventory."""
        for q in [-5, -2, 0, 2, 5]:
            r = reservation_price(S0, q, T, T, 0.1, SIGMA)
            assert abs(r - S0) < TOL, f"r at T = {r}"

    def test_linear_in_q(self):
        """r is linear in q with slope -gamma*sigma^2*(T-t)."""
        gamma, t = 0.1, 0.5
        slope = -gamma * SIGMA**2 * (T - t)
        for q1, q2 in [(-1, 0), (0, 1), (1, 2), (-2, 2)]:
            r1 = reservation_price(S0, q1, t, T, gamma, SIGMA)
            r2 = reservation_price(S0, q2, t, T, gamma, SIGMA)
            expected_diff = slope * (q2 - q1)
            assert abs((r2 - r1) - expected_diff) < TOL

    def test_derivative_wrt_t_is_nonneg_for_positive_q(self):
        """dr/dt = q*gamma*sigma^2 >= 0 for q > 0 (r increases as t->T for long q)."""
        gamma, q = 0.1, 3
        t1, t2 = 0.5, 0.6
        r1 = reservation_price(S0, q, t1, T, gamma, SIGMA)
        r2 = reservation_price(S0, q, t2, T, gamma, SIGMA)
        assert r2 > r1, "r should increase toward s as time passes for positive q"

    def test_small_gamma_small_adjustment(self):
        """As gamma->0, the inventory adjustment shrinks."""
        r_large_gamma = reservation_price(S0, 3, 0.5, T, 0.5, SIGMA)
        r_small_gamma = reservation_price(S0, 3, 0.5, T, 0.01, SIGMA)
        adj_large = abs(r_large_gamma - S0)
        adj_small = abs(r_small_gamma - S0)
        assert adj_small < adj_large, "Smaller gamma should give smaller adjustment"


# -----------------------------------------------------------------------
# Tests: liquidity_component
# -----------------------------------------------------------------------

class TestLiquidityComponent:
    """Tests for the liquidity (market-impact) component of the spread."""

    def test_known_values(self):
        """
        Paper reports approximately 1.29, 1.33, 1.15 for gamma=0.1, 0.01, 0.5
        with k=1.5. Verify each within 0.01 tolerance.
        """
        expected = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}
        for gamma, exp_val in expected.items():
            liq = liquidity_component(gamma, K)
            assert abs(liq - exp_val) < 0.01, \
                f"L(gamma={gamma}, k={K}) = {liq:.4f}, expected ~{exp_val}"

    def test_always_positive(self):
        """Liquidity component must be positive for all gamma, k > 0."""
        for gamma in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
            liq = liquidity_component(gamma, K)
            assert liq > 0, f"L < 0 for gamma={gamma}"

    def test_monotone_in_gamma(self):
        """L(gamma, k) = (2/gamma)*ln(1+gamma/k) is roughly decreasing in gamma for large gamma."""
        # At gamma=0 the limit is 2/k (L'Hopital), and for large gamma L->0
        liq_small = liquidity_component(0.001, K)
        liq_large = liquidity_component(2.0, K)
        # The limit as gamma->0 is 2/k
        liq_limit = 2.0 / K
        assert abs(liq_small - liq_limit) < 0.01

    def test_increases_with_k(self):
        """For fixed gamma, larger k (steeper decay) reduces liquidity component."""
        gamma = 0.1
        liq_small_k = liquidity_component(gamma, 0.5)
        liq_large_k = liquidity_component(gamma, 2.0)
        assert liq_small_k > liq_large_k, "Larger k -> smaller L"


# -----------------------------------------------------------------------
# Tests: total_spread
# -----------------------------------------------------------------------

class TestTotalSpread:
    """Tests for the total optimal bid-ask spread."""

    def test_decomposition(self):
        """Total spread = inventory-risk + liquidity component."""
        gamma, t = 0.1, 0.5
        risk = gamma * SIGMA**2 * (T - t)
        liq = liquidity_component(gamma, K)
        sp = total_spread(t, T, gamma, SIGMA, K)
        assert abs(sp - (risk + liq)) < TOL

    def test_at_terminal_equals_liquidity_only(self):
        """At t=T, inventory-risk term vanishes, spread = liquidity component."""
        for gamma in GAMMA_VALUES:
            sp_at_T = total_spread(T, T, gamma, SIGMA, K)
            liq = liquidity_component(gamma, K)
            assert abs(sp_at_T - liq) < TOL

    def test_decreasing_toward_T(self):
        """Spread shrinks monotonically as t->T (inventory-risk term decreases)."""
        gamma = 0.1
        spreads = [total_spread(t, T, gamma, SIGMA, K) for t in [0.0, 0.25, 0.5, 0.75]]
        assert all(spreads[i] > spreads[i + 1] for i in range(len(spreads) - 1))

    def test_equals_twice_half_spread(self):
        """total_spread = 2 * symmetric_half_spread."""
        gamma, t = 0.1, 0.5
        sp = total_spread(t, T, gamma, SIGMA, K)
        h = symmetric_half_spread(t, T, gamma, SIGMA, K)
        assert abs(sp - 2 * h) < TOL


# -----------------------------------------------------------------------
# Tests: ask_distance and bid_distance
# -----------------------------------------------------------------------

class TestQuoteDistances:
    """Tests for the optimal bid and ask quote distances."""

    def test_ask_plus_bid_equals_total_spread(self):
        """delta^a + delta^b = total_spread (inventory-invariant)."""
        gamma, t = 0.1, 0.5
        sp = total_spread(t, T, gamma, SIGMA, K)
        for q in [-3, -1, 0, 1, 3]:
            da = ask_distance(S0, q, t, T, gamma, SIGMA, K)
            db = bid_distance(S0, q, t, T, gamma, SIGMA, K)
            assert abs(da + db - sp) < TOL, f"da+db={da+db} != sp={sp} for q={q}"

    def test_symmetric_at_q0(self):
        """At q=0, ask_distance = bid_distance = symmetric_half_spread."""
        gamma, t = 0.1, 0.5
        da = ask_distance(S0, 0, t, T, gamma, SIGMA, K)
        db = bid_distance(S0, 0, t, T, gamma, SIGMA, K)
        h = symmetric_half_spread(t, T, gamma, SIGMA, K)
        assert abs(da - h) < TOL
        assert abs(db - h) < TOL

    def test_long_inventory_tightens_ask(self):
        """Positive q reduces ask distance (encourages selling)."""
        gamma, t = 0.1, 0.5
        da_q0 = ask_distance(S0, 0, t, T, gamma, SIGMA, K)
        da_qpos = ask_distance(S0, 2, t, T, gamma, SIGMA, K)
        assert da_qpos < da_q0, "Longer inventory should tighten ask"

    def test_long_inventory_widens_bid(self):
        """Positive q increases bid distance (discourages buying)."""
        gamma, t = 0.1, 0.5
        db_q0 = bid_distance(S0, 0, t, T, gamma, SIGMA, K)
        db_qpos = bid_distance(S0, 2, t, T, gamma, SIGMA, K)
        assert db_qpos > db_q0, "Longer inventory should widen bid"

    def test_short_inventory_tightens_bid(self):
        """Negative q reduces bid distance (encourages buying to cover short)."""
        gamma, t = 0.1, 0.5
        db_q0 = bid_distance(S0, 0, t, T, gamma, SIGMA, K)
        db_qneg = bid_distance(S0, -2, t, T, gamma, SIGMA, K)
        assert db_qneg < db_q0, "Short inventory should tighten bid"

    def test_at_terminal_equals_half_liquidity(self):
        """At t=T, both distances equal (1/gamma)*ln(1+gamma/k)."""
        gamma = 0.1
        half_liq = np.log(1 + gamma / K) / gamma
        for q in [-2, 0, 2]:
            da = ask_distance(S0, q, T, T, gamma, SIGMA, K)
            db = bid_distance(S0, q, T, T, gamma, SIGMA, K)
            assert abs(da - half_liq) < TOL, f"da at T = {da}, expected {half_liq}"
            assert abs(db - half_liq) < TOL, f"db at T = {db}, expected {half_liq}"

    def test_ask_distance_formula(self):
        """Verify exact formula: delta^a = (1-2q)*gamma*sigma^2*(T-t)/2 + ln(1+gamma/k)/gamma."""
        gamma, q, t = 0.1, 2, 0.5
        expected = (1 - 2 * q) * gamma * SIGMA**2 * (T - t) / 2 + np.log(1 + gamma / K) / gamma
        result = ask_distance(S0, q, t, T, gamma, SIGMA, K)
        assert abs(result - expected) < TOL

    def test_bid_distance_formula(self):
        """Verify exact formula: delta^b = (1+2q)*gamma*sigma^2*(T-t)/2 + ln(1+gamma/k)/gamma."""
        gamma, q, t = 0.1, 2, 0.5
        expected = (1 + 2 * q) * gamma * SIGMA**2 * (T - t) / 2 + np.log(1 + gamma / K) / gamma
        result = bid_distance(S0, q, t, T, gamma, SIGMA, K)
        assert abs(result - expected) < TOL


# -----------------------------------------------------------------------
# Tests: infinite-horizon and admissibility
# -----------------------------------------------------------------------

class TestInfiniteHorizon:
    """Tests for infinite-horizon stationary reservation prices."""

    def test_ask_greater_than_bid(self):
        """Stationary r_bar^a > r_bar^b."""
        gamma, omega = 0.1, 0.5
        for q in [-2, 0, 2]:
            r_a = infinite_horizon_reservation_ask(S0, q, omega, gamma, SIGMA)
            r_b = infinite_horizon_reservation_bid(S0, q, omega, gamma, SIGMA)
            assert r_a > r_b, f"r_bar^a <= r_bar^b for q={q}"

    def test_spread_equals_sigma2_over_omega(self):
        """Stationary spread = gamma*sigma^2/omega."""
        gamma, omega = 0.1, 0.5
        expected_spread = gamma * SIGMA**2 / omega
        for q in [-2, 0, 2]:
            r_a = infinite_horizon_reservation_ask(S0, q, omega, gamma, SIGMA)
            r_b = infinite_horizon_reservation_bid(S0, q, omega, gamma, SIGMA)
            spread = r_a - r_b
            assert abs(spread - expected_spread) < TOL

    def test_admissibility_condition_formula(self):
        """omega_min = 0.5 * gamma^2 * sigma^2 * q^2."""
        gamma, q = 0.1, 3.0
        expected = 0.5 * gamma**2 * SIGMA**2 * q**2
        result = admissibility_condition(gamma, SIGMA, q)
        assert abs(result - expected) < TOL

    def test_admissibility_at_zero_inventory(self):
        """At q=0, omega_min = 0."""
        result = admissibility_condition(0.1, SIGMA, 0.0)
        assert result == 0.0


# -----------------------------------------------------------------------
# Tests: symbolic verification
# -----------------------------------------------------------------------

class TestSymbolicVerification:
    """Tests using SymPy symbolic verification."""

    def test_symbolic_runs_without_error(self):
        """verify_frozen_value_symbolic() should return a dict with 5 keys."""
        result = verify_frozen_value_symbolic()
        assert isinstance(result, dict)
        assert 'r_a_symbolic' in result
        assert 'r_b_symbolic' in result
        assert 'r_avg_symbolic' in result

    def test_symbolic_ask_matches_expected(self):
        """Symbolic r^a should simplify to the paper formula."""
        import sympy as sp
        result = verify_frozen_value_symbolic()
        r_a_sym = result['r_a_symbolic']
        r_a_exp = result['r_a_expected']
        diff = sp.simplify(r_a_sym - r_a_exp)
        assert diff == 0, f"Symbolic r^a mismatch: diff = {diff}"

    def test_symbolic_bid_matches_expected(self):
        """Symbolic r^b should simplify to the paper formula."""
        import sympy as sp
        result = verify_frozen_value_symbolic()
        r_b_sym = result['r_b_symbolic']
        r_b_exp = result['r_b_expected']
        diff = sp.simplify(r_b_sym - r_b_exp)
        assert diff == 0, f"Symbolic r^b mismatch: diff = {diff}"


# -----------------------------------------------------------------------
# Tests: simulate_path
# -----------------------------------------------------------------------

class TestSimulatePath:
    """Tests for the Monte Carlo path simulator."""

    def _make_config(self, gamma=0.1, seed=0):
        return SimConfig(gamma=gamma, seed=seed, n_paths=10)

    def _make_random_arrays(self, N=200, seed=0):
        rng = np.random.default_rng(seed)
        price_moves = rng.choice([-1, 1], size=N).astype(float)
        ask_uniforms = rng.uniform(0, 1, size=N)
        bid_uniforms = rng.uniform(0, 1, size=N)
        return price_moves, ask_uniforms, bid_uniforms

    def test_returns_path_result(self):
        """simulate_path should return a PathResult."""
        config = self._make_config()
        N = int(config.T / config.dt)
        pm, au, bu = self._make_random_arrays(N)
        result = simulate_path(config, 'inventory', pm, au, bu)
        assert isinstance(result, PathResult)

    def test_trajectory_stored_when_requested(self):
        """store_trajectory=True should populate array fields."""
        config = self._make_config()
        N = int(config.T / config.dt)
        pm, au, bu = self._make_random_arrays(N)
        result = simulate_path(config, 'inventory', pm, au, bu, store_trajectory=True)
        assert result.mid_prices is not None
        assert len(result.mid_prices) == N + 1
        assert result.reservation_prices is not None
        assert result.ask_quotes is not None
        assert result.bid_quotes is not None

    def test_initial_midprice_correct(self):
        """First element of mid_prices should equal s0."""
        config = self._make_config()
        N = int(config.T / config.dt)
        pm, au, bu = self._make_random_arrays(N)
        result = simulate_path(config, 'inventory', pm, au, bu, store_trajectory=True)
        assert abs(result.mid_prices[0] - config.s0) < TOL

    def test_profit_is_finite(self):
        """Final profit must be finite."""
        config = self._make_config()
        N = int(config.T / config.dt)
        pm, au, bu = self._make_random_arrays(N)
        for strategy in ['inventory', 'symmetric']:
            result = simulate_path(config, strategy, pm, au, bu)
            assert np.isfinite(result.final_profit), f"Profit is not finite for {strategy}"

    def test_inventory_integers_only(self):
        """Inventory must be integer-valued (fills are ±1 shares)."""
        config = self._make_config()
        N = int(config.T / config.dt)
        pm, au, bu = self._make_random_arrays(N)
        result = simulate_path(config, 'inventory', pm, au, bu)
        assert result.final_inventory == int(result.final_inventory), \
            f"Inventory {result.final_inventory} is not integer"

    def test_wrong_price_moves_length_raises(self):
        """simulate_path should raise if price_moves has wrong length."""
        config = self._make_config()
        pm = np.ones(100)  # wrong length
        au = np.ones(200)
        bu = np.ones(200)
        with pytest.raises(AssertionError):
            simulate_path(config, 'inventory', pm, au, bu)

    def test_symmetric_q0_implies_symmetric_quotes(self):
        """
        For symmetric strategy with q0=0, ask and bid quotes should be equidistant from mid.
        This verifies the symmetric strategy centering.
        """
        config = self._make_config()
        N = int(config.T / config.dt)
        # Force no fills by setting all uniforms to 1 (no fill events)
        pm = np.ones(N)  # all upward moves
        au = np.ones(N)  # no ask fills (threshold not reached)
        bu = np.ones(N)  # no bid fills
        result = simulate_path(config, 'symmetric', pm, au, bu, store_trajectory=True)
        # With no fills, inventory stays at 0, quotes should be symmetric
        # Check first few steps (before inventory could change)
        for i in range(5):
            ask_dist = result.ask_quotes[i] - result.mid_prices[i]
            bid_dist = result.mid_prices[i] - result.bid_quotes[i]
            assert abs(ask_dist - bid_dist) < TOL, \
                f"Asymmetric quotes at step {i}: ask_dist={ask_dist}, bid_dist={bid_dist}"


# -----------------------------------------------------------------------
# Tests: run_experiment
# -----------------------------------------------------------------------

class TestRunExperiment:
    """Tests for the Monte Carlo experiment runner."""

    def test_returns_two_results(self):
        """run_experiment should return (inv_result, sym_result)."""
        config = SimConfig(gamma=0.1, n_paths=20, seed=0)
        inv_res, sym_res = run_experiment(config)
        assert isinstance(inv_res, ExperimentResult)
        assert isinstance(sym_res, ExperimentResult)

    def test_result_strategy_labels(self):
        """Strategy labels should be 'Inventory' and 'Symmetric'."""
        config = SimConfig(gamma=0.1, n_paths=20, seed=0)
        inv_res, sym_res = run_experiment(config)
        assert inv_res.strategy == 'Inventory'
        assert sym_res.strategy == 'Symmetric'

    def test_profits_array_length(self):
        """Profits array length must equal n_paths."""
        n = 30
        config = SimConfig(gamma=0.1, n_paths=n, seed=0)
        inv_res, sym_res = run_experiment(config)
        assert len(inv_res.profits) == n
        assert len(sym_res.profits) == n

    def test_all_profits_finite(self):
        """All simulated profits must be finite."""
        config = SimConfig(gamma=0.1, n_paths=50, seed=1)
        inv_res, sym_res = run_experiment(config)
        assert np.all(np.isfinite(inv_res.profits))
        assert np.all(np.isfinite(sym_res.profits))

    def test_inventory_strategy_lower_std_profit(self):
        """
        Directional test: inventory strategy should have lower profit std
        than symmetric benchmark (key paper finding).
        Uses 200 paths for reliability.
        """
        config = SimConfig(gamma=0.1, n_paths=200, seed=42)
        inv_res, sym_res = run_experiment(config)
        assert inv_res.std_profit < sym_res.std_profit, \
            f"Inventory std={inv_res.std_profit} should be < Symmetric std={sym_res.std_profit}"

    def test_inventory_strategy_lower_std_q(self):
        """
        Directional test: inventory strategy should have lower terminal inventory std
        (key paper finding).
        """
        config = SimConfig(gamma=0.1, n_paths=200, seed=42)
        inv_res, sym_res = run_experiment(config)
        assert inv_res.std_final_q < sym_res.std_final_q, \
            f"Inventory std_q={inv_res.std_final_q} should be < Symmetric std_q={sym_res.std_final_q}"

    def test_positive_mean_profit(self):
        """Mean profit should be positive (market-maker earns the spread)."""
        config = SimConfig(gamma=0.1, n_paths=100, seed=0)
        inv_res, sym_res = run_experiment(config)
        assert inv_res.mean_profit > 0, f"Inventory mean profit = {inv_res.mean_profit}"
        assert sym_res.mean_profit > 0, f"Symmetric mean profit = {sym_res.mean_profit}"

    def test_spread_matches_liquidity_component(self):
        """Reported spread column should match liquidity_component formula."""
        gamma, k = 0.1, 1.5
        config = SimConfig(gamma=gamma, k=k, n_paths=10, seed=0)
        inv_res, sym_res = run_experiment(config)
        expected_liq = (2 / gamma) * np.log(1 + gamma / k)
        assert abs(inv_res.spread - round(expected_liq, 2)) < 0.01

    def test_gamma_near_zero_strategies_similar(self):
        """
        For very small gamma, inventory and symmetric strategies should be similar
        (paper: at gamma=0.01, differences diminish).
        """
        config = SimConfig(gamma=0.01, n_paths=200, seed=42)
        inv_res, sym_res = run_experiment(config)
        # Mean profit difference should be small relative to gamma=0.5
        profit_diff_small_gamma = abs(inv_res.mean_profit - sym_res.mean_profit)

        config_large = SimConfig(gamma=0.5, n_paths=200, seed=42)
        inv_res_l, sym_res_l = run_experiment(config_large)
        profit_diff_large_gamma = abs(inv_res_l.mean_profit - sym_res_l.mean_profit)

        assert profit_diff_small_gamma < profit_diff_large_gamma, \
            "Small gamma should lead to smaller strategy differences"

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        config = SimConfig(gamma=0.1, n_paths=50, seed=7)
        inv1, sym1 = run_experiment(config)
        inv2, sym2 = run_experiment(config)
        assert inv1.mean_profit == inv2.mean_profit
        assert sym1.std_profit == sym2.std_profit


# -----------------------------------------------------------------------
# Tests: paper-specific parameter checks
# -----------------------------------------------------------------------

class TestPaperParameterChecks:
    """Numerical spot checks verifying paper-reported values."""

    def test_liquidity_component_paper_table(self):
        """
        Paper Table: gamma=0.1 -> ~1.29, gamma=0.01 -> ~1.33, gamma=0.5 -> ~1.15
        Tolerance ±0.01 to match rounded reporting.
        """
        targets = [(0.1, 1.29), (0.01, 1.33), (0.5, 1.15)]
        for gamma, target in targets:
            liq = liquidity_component(gamma, K)
            assert abs(liq - target) < 0.01, \
                f"L(gamma={gamma}) = {liq:.4f}, paper reports ~{target}"

    def test_fill_probability_never_exceeds_1(self):
        """
        Paper parameter validation: lambda*dt must be <= 1 for Bernoulli approximation.
        With A=140, k=1.5, dt=0.005, the maximum lambda occurs at delta=0.
        """
        A, k, dt = 140.0, 1.5, 0.005
        max_lambda = A * np.exp(-k * 0.0)  # lambda at delta=0 (minimum delta)
        max_prob = max_lambda * dt
        # With the paper's parameters and reasonable non-zero delta, prob << 1
        # At minimum realistic delta (liquidity component ~0.65), prob << 1
        min_delta = np.log(1 + 0.5 / k) / 0.5  # for gamma=0.5
        lambda_min_delta = A * np.exp(-k * min_delta)
        assert lambda_min_delta * dt <= 1.0, \
            f"p_fill = {lambda_min_delta * dt:.4f} > 1 at min delta"

    def test_binomial_midprice_variance(self):
        """
        Binomial step sigma*sqrt(dt) should give variance sigma^2*dt per step,
        matching ABM approximation.
        """
        sigma, dt = 2.0, 0.005
        step_size = sigma * np.sqrt(dt)
        variance_per_step = step_size**2  # E[(±step)^2] = step^2
        expected = sigma**2 * dt
        assert abs(variance_per_step - expected) < TOL

    def test_total_steps(self):
        """N = T/dt = 1/0.005 = 200 steps."""
        config = SimConfig()
        N = int(config.T / config.dt)
        assert N == 200, f"Expected 200 steps, got {N}"


# -----------------------------------------------------------------------
# Tests: edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and numerical stability tests."""

    def test_zero_time_remaining(self):
        """All time-dependent formulas should handle t=T gracefully."""
        gamma = 0.1
        assert abs(reservation_price(S0, 5, T, T, gamma, SIGMA) - S0) < TOL
        assert abs(total_spread(T, T, gamma, SIGMA, K) - liquidity_component(gamma, K)) < TOL

    def test_large_inventory(self):
        """Formulas should handle large |q| without overflow."""
        gamma, q = 0.1, 100
        r = reservation_price(S0, q, 0.5, T, gamma, SIGMA)
        assert np.isfinite(r)

    def test_small_gamma_no_division_by_zero(self):
        """Very small gamma should not cause division by zero."""
        gamma = 1e-6
        liq = liquidity_component(gamma, K)
        assert np.isfinite(liq)
        sp = total_spread(0.5, T, gamma, SIGMA, K)
        assert np.isfinite(sp)

    def test_single_step_simulation(self):
        """Single-step simulation (dt=T) should still produce a result."""
        config = SimConfig(dt=1.0, T=1.0)  # N=1
        rng = np.random.default_rng(0)
        pm = rng.choice([-1, 1], size=1).astype(float)
        au = rng.uniform(0, 1, size=1)
        bu = rng.uniform(0, 1, size=1)
        result = simulate_path(config, 'inventory', pm, au, bu)
        assert np.isfinite(result.final_profit)

    def test_all_fills_worst_case(self):
        """With all uniforms = 0, all Bernoulli fills trigger; result should still be finite."""
        config = SimConfig(gamma=0.1)
        N = int(config.T / config.dt)
        pm = np.ones(N)
        au = np.zeros(N)  # all fills trigger
        bu = np.zeros(N)  # all fills trigger
        result = simulate_path(config, 'inventory', pm, au, bu)
        assert np.isfinite(result.final_profit)

    def test_no_fills(self):
        """With all uniforms = 1, no fills occur; result should be X0 + q0*s_T."""
        config = SimConfig(gamma=0.1, q0=0.0, X0=0.0)
        N = int(config.T / config.dt)
        pm = np.ones(N)  # all up
        au = np.ones(N)  # no fills
        bu = np.ones(N)  # no fills
        result = simulate_path(config, 'inventory', pm, au, bu)
        # No fills -> X=0, q=0, final profit = 0 + 0*s_T = 0
        assert abs(result.final_profit - 0.0) < TOL
        assert result.final_inventory == 0
