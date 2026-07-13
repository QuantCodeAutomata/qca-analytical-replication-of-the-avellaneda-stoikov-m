"""
Avellaneda-Stoikov Market-Making Model: Analytical Formulas

Reference: Avellaneda & Stoikov, 'High-frequency trading in a limit order book',
Quantitative Finance, 2008.

This module implements the closed-form analytical expressions derived in the paper,
including frozen-inventory reservation prices, CARA utility transformation,
first-order optimality conditions, and the approximate explicit simulation formulas.
"""
import numpy as np
import sympy as sp
from typing import Union


def frozen_value_function(x: float, s: float, q: float, t: float, T: float,
                          gamma: float, sigma: float) -> float:
    """
    Frozen-inventory utility value function under CARA utility and ABM mid-price.

    Derived by treating inventory q as constant and solving the resulting
    one-dimensional HJB equation. Under CARA utility U(W) = -exp(-gamma*W)
    and an arithmetic Brownian motion for the mid-price, the value function
    separates multiplicatively in x, q*s, and a quadratic variance term.

    Formula:
        v(x, s, q, t) = -exp(-gamma*x)
                        * exp(-gamma*q*s)
                        * exp(gamma^2 * q^2 * sigma^2 * (T - t) / 2)

    The three exponential factors represent, respectively:
        - The CARA penalty on cash wealth.
        - The mark-to-market value of the inventory position.
        - The variance cost of holding q units over the remaining horizon (T-t).

    Args:
        x:     Current cash (numeraire) balance.
        s:     Current mid-price of the risky asset.
        q:     Current inventory (number of units held; may be negative).
        t:     Current time.
        T:     Terminal horizon (must satisfy T >= t).
        gamma: Risk-aversion coefficient (CARA parameter; must be > 0).
        sigma: Volatility of the mid-price diffusion (must be > 0).

    Returns:
        Utility value v(x, s, q, t), which is always strictly negative.
    """
    tau = T - t  # time remaining to terminal horizon
    return (
        -np.exp(-gamma * x)
        * np.exp(-gamma * q * s)
        * np.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


def reservation_ask_frozen(s: float, q: float, t: float, T: float,
                           gamma: float, sigma: float) -> float:
    """
    Frozen-inventory reservation ask price derived from utility indifference.

    The reservation ask r^a is the minimum price at which the agent is willing
    to sell one unit of inventory, leaving utility unchanged:

        v(x + r^a, s, q - 1, t) = v(x, s, q, t)

    Solving this indifference equation under the frozen-inventory value function
    yields the closed-form expression:

        r^a(s, q, t) = s + ((1 - 2*q) * gamma * sigma^2 * (T - t)) / 2

    Interpretation:
        - The term ``s`` anchors the ask to the current mid-price.
        - The correction ``(1 - 2*q) * gamma * sigma^2 * (T - t) / 2`` reflects
          the risk cost of carrying q units: a long inventory (q > 0) lowers the
          ask, incentivising faster liquidation.

    Args:
        s:     Current mid-price.
        q:     Current inventory.
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).

    Returns:
        Frozen-inventory reservation ask price r^a(s, q, t).
    """
    tau = T - t
    return s + (1 - 2 * q) * gamma * sigma**2 * tau / 2


def reservation_bid_frozen(s: float, q: float, t: float, T: float,
                           gamma: float, sigma: float) -> float:
    """
    Frozen-inventory reservation bid price derived from utility indifference.

    The reservation bid r^b is the maximum price the agent is willing to pay
    to buy one additional unit, leaving utility unchanged:

        v(x - r^b, s, q + 1, t) = v(x, s, q, t)

    Solving under the frozen-inventory value function gives:

        r^b(s, q, t) = s + ((-1 - 2*q) * gamma * sigma^2 * (T - t)) / 2

    Interpretation:
        - For zero inventory (q = 0) the bid lies below the mid-price by
          ``gamma * sigma^2 * (T - t) / 2``.
        - A long inventory (q > 0) depresses the bid further, reflecting the
          cost of adding to an already risky position.

    Args:
        s:     Current mid-price.
        q:     Current inventory.
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).

    Returns:
        Frozen-inventory reservation bid price r^b(s, q, t).
    """
    tau = T - t
    return s + (-1 - 2 * q) * gamma * sigma**2 * tau / 2


def reservation_price(s: float, q: float, t: float, T: float,
                      gamma: float, sigma: float) -> float:
    """
    Reservation (indifference) price: arithmetic mean of frozen reservation ask and bid.

    Taking the midpoint of r^a and r^b cancels the constant spread component,
    leaving the inventory-adjusted mid-price used operationally in the paper's
    simulation:

        r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t)

    This is the price at which the agent is indifferent to buying or selling an
    infinitesimal quantity. It equals the mid-price discounted by the risk cost
    of the current inventory over the remaining horizon.

    Args:
        s:     Current mid-price.
        q:     Current inventory.
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).

    Returns:
        Reservation price r(s, q, t).
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


def liquidity_component(gamma: float, k: float) -> float:
    """
    Liquidity (market-impact) component of the optimal bid-ask spread.

    Arises from the first-order condition on the order-arrival intensity.
    When the Poisson arrival rate decays exponentially with quote distance
    (lambda(delta) = A * exp(-k * delta)), the optimal static contribution
    to each half-spread is:

        L(gamma, k) = (2 / gamma) * ln(1 + gamma / k)

    This term is independent of inventory and remaining time; it captures the
    trade-off between earning a wider spread and the resulting reduction in
    fill probability.

    Args:
        gamma: Risk-aversion coefficient (> 0).
        k:     Exponential decay rate of the arrival intensity (> 0).

    Returns:
        Liquidity component L(gamma, k) >= 0.
    """
    return (2 / gamma) * np.log(1 + gamma / k)


def total_spread(t: float, T: float, gamma: float, sigma: float, k: float) -> float:
    """
    Total optimal bid-ask spread: inventory-risk component plus liquidity component.

    The optimal spread decomposes additively into:
        1. An inventory-risk (variance) term: ``gamma * sigma^2 * (T - t)``
        2. A liquidity (market-impact) term: ``(2 / gamma) * ln(1 + gamma / k)``

    Full expression:

        spread(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    At terminal time (t = T) the risk component vanishes and only the liquidity
    component remains. The spread widens as the remaining horizon grows because
    the agent needs more compensation for inventory risk.

    Args:
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).
        k:     Exponential decay rate of the order-arrival intensity (> 0).

    Returns:
        Total optimal spread spread(t) >= 0.
    """
    tau = T - t
    risk_component = gamma * sigma**2 * tau
    liquidity = liquidity_component(gamma, k)
    return risk_component + liquidity


def ask_distance(s: float, q: float, t: float, T: float,
                 gamma: float, sigma: float, k: float) -> float:
    """
    Optimal ask quote distance from the mid-price (delta^a).

    The ask quote is placed at s + delta^a, where delta^a solves the
    first-order condition for the agent's HJB equation. It equals the
    sum of the ask's inventory-adjusted reservation premium and the
    half of the liquidity component:

        delta^a(s, q, t) = ((1 - 2*q) * gamma * sigma^2 * (T - t)) / 2
                           + (1 / gamma) * ln(1 + gamma / k)

    Equivalently, delta^a = (r^a - s) + L / 2, where L is the liquidity
    component and r^a is the frozen reservation ask.

    Interpretation:
        - When q > 0 (long inventory) the agent tightens the ask to accelerate
          selling, since (1 - 2*q) decreases.
        - When q < 0 (short inventory) the ask is widened to discourage further
          short positions.

    Args:
        s:     Current mid-price (not used in the computation but kept for
               symmetry with other functions in the API).
        q:     Current inventory.
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).
        k:     Exponential decay rate of the order-arrival intensity (> 0).

    Returns:
        Ask distance delta^a >= 0 (distance from mid-price to optimal ask quote).
    """
    tau = T - t
    # Inventory-adjusted reservation component for the ask side
    reservation_component = (1 - 2 * q) * gamma * sigma**2 * tau / 2
    # Half the liquidity component (symmetric across bid and ask)
    half_liquidity = np.log(1 + gamma / k) / gamma
    return reservation_component + half_liquidity


def bid_distance(s: float, q: float, t: float, T: float,
                 gamma: float, sigma: float, k: float) -> float:
    """
    Optimal bid quote distance from the mid-price (delta^b).

    The bid quote is placed at s - delta^b, where delta^b solves the
    first-order condition for the agent's HJB equation:

        delta^b(s, q, t) = ((1 + 2*q) * gamma * sigma^2 * (T - t)) / 2
                           + (1 / gamma) * ln(1 + gamma / k)

    Equivalently, delta^b = (s - r^b) + L / 2, where L is the liquidity
    component and r^b is the frozen reservation bid.

    Interpretation:
        - When q > 0 (long inventory) the agent widens the bid to discourage
          further buying, since (1 + 2*q) increases.
        - When q < 0 (short inventory) the bid is tightened to accelerate
          covering the short position.

    Args:
        s:     Current mid-price (not used in the computation but kept for
               symmetry with other functions in the API).
        q:     Current inventory.
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).
        k:     Exponential decay rate of the order-arrival intensity (> 0).

    Returns:
        Bid distance delta^b >= 0 (distance from mid-price to optimal bid quote).
    """
    tau = T - t
    # Inventory-adjusted reservation component for the bid side
    reservation_component = (1 + 2 * q) * gamma * sigma**2 * tau / 2
    # Half the liquidity component (symmetric across bid and ask)
    half_liquidity = np.log(1 + gamma / k) / gamma
    return reservation_component + half_liquidity


def symmetric_half_spread(t: float, T: float, gamma: float, sigma: float, k: float) -> float:
    """
    Symmetric benchmark strategy half-spread (special case q = 0).

    When inventory is zero the ask and bid distances are equal, yielding a
    symmetric spread centred on the mid-price. The common half-spread is:

        h(t) = (gamma * sigma^2 * (T - t)) / 2 + (1 / gamma) * ln(1 + gamma / k)

    This is obtained by setting q = 0 in either ``ask_distance`` or
    ``bid_distance``. It serves as a benchmark for the inventory-neutral agent
    and corresponds to total_spread(t) / 2.

    Args:
        t:     Current time.
        T:     Terminal horizon (T >= t).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).
        k:     Exponential decay rate of the order-arrival intensity (> 0).

    Returns:
        Symmetric half-spread h(t) = total_spread(t) / 2.
    """
    tau = T - t
    risk_half = gamma * sigma**2 * tau / 2
    half_liquidity = np.log(1 + gamma / k) / gamma
    return risk_half + half_liquidity


def infinite_horizon_reservation_ask(s: float, q: float, omega: float,
                                      gamma: float, sigma: float) -> float:
    """
    Infinite-horizon stationary reservation ask price.

    In the infinite-horizon formulation the agent discounts future utility at
    rate omega > 0. When omega is large enough (see ``admissibility_condition``),
    a stationary solution exists. In the simplified linear approximation the
    stationary reservation ask is:

        rbar^a(s, q) = s - q * gamma * sigma^2 / omega
                       + gamma * sigma^2 / (2 * omega)

    The term ``-q * gamma * sigma^2 / omega`` is the inventory-risk adjustment
    (positive when short, negative when long). The additive term
    ``gamma * sigma^2 / (2 * omega)`` represents the ask-side premium relative
    to the symmetric reservation price.

    Admissibility requires: omega > 0.5 * gamma^2 * sigma^2 * q^2
    (see ``admissibility_condition``).

    Args:
        s:     Current mid-price.
        q:     Current inventory.
        omega: Discount rate for the infinite-horizon problem (> 0).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).

    Returns:
        Stationary reservation ask price rbar^a(s, q).
    """
    # Inventory-risk discount relative to the mid-price
    inventory_adjustment = -q * gamma * sigma**2 / omega
    # Ask-side premium (half the stationary spread)
    ask_premium = gamma * sigma**2 / (2 * omega)
    return s + inventory_adjustment + ask_premium


def infinite_horizon_reservation_bid(s: float, q: float, omega: float,
                                      gamma: float, sigma: float) -> float:
    """
    Infinite-horizon stationary reservation bid price.

    Symmetric counterpart to ``infinite_horizon_reservation_ask``. In the
    simplified linear approximation:

        rbar^b(s, q) = s - q * gamma * sigma^2 / omega
                       - gamma * sigma^2 / (2 * omega)

    The bid price lies below the stationary reservation ask by the full
    stationary spread ``gamma * sigma^2 / omega``.

    Admissibility requires: omega > 0.5 * gamma^2 * sigma^2 * q^2
    (see ``admissibility_condition``).

    Args:
        s:     Current mid-price.
        q:     Current inventory.
        omega: Discount rate for the infinite-horizon problem (> 0).
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).

    Returns:
        Stationary reservation bid price rbar^b(s, q).
    """
    # Inventory-risk discount relative to the mid-price
    inventory_adjustment = -q * gamma * sigma**2 / omega
    # Bid-side discount (half the stationary spread, subtracted)
    bid_discount = gamma * sigma**2 / (2 * omega)
    return s + inventory_adjustment - bid_discount


def admissibility_condition(gamma: float, sigma: float, q: float) -> float:
    """
    Minimum discount rate omega for admissibility of the infinite-horizon problem.

    For the infinite-horizon value function to be well-defined (i.e., for
    the discounted integral of utility to converge), the discount rate omega
    must dominate the variance cost of the inventory position:

        omega_min(q) = 0.5 * gamma^2 * sigma^2 * q^2

    In practice, to guarantee admissibility for all inventories up to some
    maximum level q_max, one sets:

        omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2

    Args:
        gamma: Risk-aversion coefficient (> 0).
        sigma: Mid-price volatility (> 0).
        q:     Inventory level at which to evaluate the minimum discount rate.

    Returns:
        omega_min(q) = minimum discount rate required for admissibility at
        inventory level q.
    """
    return 0.5 * gamma**2 * sigma**2 * q**2


def verify_frozen_value_symbolic() -> dict:
    """
    Use SymPy to symbolically verify the frozen-inventory value function derivation.

    Constructs the frozen-inventory value function v(x, s, q, t) symbolically
    and solves the utility-indifference equations for the reservation ask (r^a)
    and reservation bid (r^b). The solutions are compared against the expected
    closed-form expressions from the paper.

    Verification steps:
        1. Define v symbolically.
        2. Solve v(x + r^a, s, q - 1, t) = v(x, s, q, t) for r^a.
        3. Solve v(x - r^b, s, q + 1, t) = v(x, s, q, t) for r^b.
        4. Compute the arithmetic mean r_avg = (r^a + r^b) / 2.

    Returns:
        A dictionary with keys:
            'r_a_symbolic'  : SymPy expression for the solved reservation ask.
            'r_b_symbolic'  : SymPy expression for the solved reservation bid.
            'r_avg_symbolic': SymPy expression for (r^a + r^b) / 2.
            'r_a_expected'  : Expected r^a from the paper's formula.
            'r_b_expected'  : Expected r^b from the paper's formula.
    """
    # Define positive symbols matching the model parameters
    gamma_s, q_s, s_s, x_s, sigma_s, T_s, t_s = sp.symbols(
        'gamma q s x sigma T t', positive=True
    )

    tau = T_s - t_s  # time remaining to terminal horizon

    # Frozen-inventory value function: v = -exp(-γx) * exp(-γqs) * exp(γ²q²σ²τ/2)
    v = -sp.exp(-gamma_s * x_s) * sp.exp(-gamma_s * q_s * s_s) * sp.exp(
        sp.Rational(1, 2) * gamma_s**2 * q_s**2 * sigma_s**2 * tau
    )

    # --- Reservation ask: solve v(x + r_a, s, q - 1, t) = v(x, s, q, t) ---
    r_a = sp.Symbol('r_a')
    # Substitute cash x -> x + r_a and inventory q -> q - 1
    v_after_ask = v.subs([(x_s, x_s + r_a), (q_s, q_s - 1)])
    # Indifference condition: ratio must equal 1, so ratio - 1 = 0
    eq_ask = sp.simplify(v_after_ask / v - 1)
    r_a_sol = sp.solve(eq_ask, r_a)[0]

    # --- Reservation bid: solve v(x - r_b, s, q + 1, t) = v(x, s, q, t) ---
    r_b = sp.Symbol('r_b')
    # Substitute cash x -> x - r_b and inventory q -> q + 1
    v_after_bid = v.subs([(x_s, x_s - r_b), (q_s, q_s + 1)])
    # Indifference condition: ratio must equal 1, so ratio - 1 = 0
    eq_bid = sp.simplify(v_after_bid / v - 1)
    r_b_sol = sp.solve(eq_bid, r_b)[0]

    # Arithmetic mean of the two reservation prices
    r_avg = sp.simplify((r_a_sol + r_b_sol) / 2)

    return {
        'r_a_symbolic': r_a_sol,
        'r_b_symbolic': r_b_sol,
        'r_avg_symbolic': r_avg,
        # Expected forms from the paper for cross-checking
        'r_a_expected': s_s + (1 - 2 * q_s) * gamma_s * sigma_s**2 * tau / 2,
        'r_b_expected': s_s + (-1 - 2 * q_s) * gamma_s * sigma_s**2 * tau / 2,
    }
