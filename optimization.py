"""
Bid-price optimization for single-price balancing markets.

Implements:
    - Theorem 1 (boundary quantity): α* ∈ {0, 1}
    - Theorem 3 (first-order condition): p^{b*} = E[λ^imb | λ^BE = p^{b*}]
    - Theorem 4 (Gaussian closed form): p^{b*} = (μ₂ − β·μ₁) / (1 − β)
    - Empirical sample-average approximation (SAA)
    - Monte Carlo revenue evaluation
    - Sensitivity analysis over model parameters
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from .data_classes import MarketParameters, OptimizationResult


class BalancingMarketOptimizer:
    """
    Optimal bidding strategies for single-price balancing markets.

    Parameters
    ----------
    params : MarketParameters
        Estimated bivariate distribution parameters.
    q : float
        Normalized delivery position (default 1.0).
    data : pd.DataFrame, optional
        Historical data for empirical methods.
    """

    def __init__(
        self,
        params: MarketParameters,
        q: float = 1.0,
        data: pd.DataFrame = None,
    ):
        self.params = params
        self.q = q
        self.data = data

    # ═══════════════════════════════════════════════════════════════════
    #  Revenue function: V_{p^b, α}(X, q)
    # ═══════════════════════════════════════════════════════════════════

    def revenue(
        self,
        pb: float,
        alpha: float,
        lam_be: np.ndarray,
        lam_imb: np.ndarray,
    ) -> np.ndarray:
        """
        Per-unit revenue (vectorised over price realisations).

        V = α·(λ^BE − λ^imb)·𝟙{λ^BE ≥ p^b} + q·λ^imb
        """
        lam_be = np.atleast_1d(lam_be)
        lam_imb = np.atleast_1d(lam_imb)
        accepted = lam_be >= pb
        spread = lam_be - lam_imb
        return alpha * spread * accepted + self.q * lam_imb

    # ═══════════════════════════════════════════════════════════════════
    #  ANALYTICAL (Gaussian) methods
    # ═══════════════════════════════════════════════════════════════════

    def expected_revenue_analytical(
        self, pb: float, alpha: float = 1.0
    ) -> float:
        """
        E[V] under bivariate Gaussian assumption.

        Uses the truncated-normal identity
            E[X | X ≥ a] = μ + σ·φ(z)/(1 − Φ(z))
        where z = (a − μ)/σ.
        """
        mu1, mu2 = self.params.mu_clearing, self.params.mu_imbalance
        s1, s2 = self.params.sigma_clearing, self.params.sigma_imbalance
        rho = self.params.rho

        z = (pb - mu1) / s1
        prob_accept = 1.0 - stats.norm.cdf(z)

        if prob_accept < 1e-12:
            return self.q * mu2

        mills = stats.norm.pdf(z) / prob_accept
        cond_exp_spread = (mu1 - mu2) + (s1 - rho * s2) * mills
        return alpha * prob_accept * cond_exp_spread + self.q * mu2

    def optimal_bid_price_gaussian(self) -> float:
        """
        Closed-form optimal bid price (Theorem 4).

            p^{b*} = (μ₂ − β·μ₁) / (1 − β),   β = ρ·σ₂/σ₁

        Returns ``μ₁`` in the degenerate case β ≈ 1.
        """
        beta = self.params.beta
        mu1 = self.params.mu_clearing
        mu2 = self.params.mu_imbalance

        if abs(1.0 - beta) < 1e-10:
            return mu1

        return (mu2 - beta * mu1) / (1.0 - beta)

    def conditional_imbalance_mean(self, pb: float) -> float:
        """E[λ^imb | λ^BE = p^b] = μ₂ + β·(p^b − μ₁)."""
        return self.params.conditional_imbalance_mean(pb)

    def verify_optimality_condition(
        self, pb: float, tol: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Check the first-order condition  E[λ^imb | λ^BE = p^{b*}] = p^{b*}.

        Returns ``(is_satisfied, absolute_gap)``.
        """
        gap = abs(self.conditional_imbalance_mean(pb) - pb)
        return gap < tol, gap

    def acceptance_probability(self, pb: float) -> float:
        """P(λ^BE ≥ p^b) under Gaussian assumption."""
        z = (pb - self.params.mu_clearing) / self.params.sigma_clearing
        return 1.0 - stats.norm.cdf(z)

    def optimize_analytical(self) -> OptimizationResult:
        """
        Full analytical optimisation (Gaussian case).

        Returns ``(p^{b*}, α* = 1, E[V*])``.
        """
        t0 = time.time()
        pb_opt = self.optimal_bid_price_gaussian()
        is_optimal, _ = self.verify_optimality_condition(pb_opt)
        expected_rev = self.expected_revenue_analytical(pb_opt, alpha=1.0)
        return OptimizationResult(
            p_bid_optimal=pb_opt,
            alpha_optimal=1.0,
            expected_revenue=expected_rev,
            method="analytical_gaussian",
            success=is_optimal,
            computation_time=time.time() - t0,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  EMPIRICAL (sample-average) methods
    # ═══════════════════════════════════════════════════════════════════

    def expected_revenue_empirical(
        self,
        pb: float,
        alpha: float = 1.0,
        data: pd.DataFrame = None,
    ) -> float:
        """Sample-average expected revenue: (1/N) Σ V(Xᵢ)."""
        df = data if data is not None else self.data
        if df is None:
            raise ValueError("No data provided for empirical calculation")
        rev = self.revenue(
            pb, alpha,
            df["clearing_price"].values,
            df["imbalance_price"].values,
        )
        return float(rev.mean())

    def optimize_empirical(
        self,
        data: pd.DataFrame = None,
        pb_range: Tuple[float, float] = None,
    ) -> OptimizationResult:
        """
        Optimise bid price by maximising empirical mean revenue (SAA).
        """
        t0 = time.time()
        df = data if data is not None else self.data
        if df is None:
            raise ValueError("No data provided for empirical optimisation")

        if pb_range is None:
            mu = df["clearing_price"].mean()
            sigma = df["clearing_price"].std()
            pb_range = (
                max(mu - 3 * sigma, df["clearing_price"].min()),
                mu + 2 * sigma,
            )

        result = minimize_scalar(
            lambda pb: -self.expected_revenue_empirical(pb, 1.0, df),
            bounds=pb_range,
            method="bounded",
        )

        return OptimizationResult(
            p_bid_optimal=float(result.x),
            alpha_optimal=1.0,
            expected_revenue=float(-result.fun),
            method="empirical_SAA",
            success=bool(result.success),
            computation_time=time.time() - t0,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  MONTE CARLO revenue (for validation)
    # ═══════════════════════════════════════════════════════════════════

    def expected_revenue_mc(
        self,
        pb: float,
        alpha: float = 1.0,
        n_samples: int = 100_000,
        seed: int = 42,
    ) -> float:
        """Monte Carlo estimate of E[V] by sampling from the fitted Gaussian."""
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(
            self.params.mean_vector,
            self.params.covariance_matrix,
            size=n_samples,
        )
        rev = self.revenue(pb, alpha, samples[:, 0], samples[:, 1])
        return float(rev.mean())

    # ═══════════════════════════════════════════════════════════════════
    #  SENSITIVITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def sensitivity_analysis(
        self, param_name: str, param_range: np.ndarray
    ) -> pd.DataFrame:
        """
        Sweep one parameter and track optimal bid, revenue, and acceptance
        probability.

        Parameters
        ----------
        param_name : str
            One of ``'rho'``, ``'sigma_clearing'``, ``'sigma_imbalance'``,
            ``'mu_clearing'``, ``'mu_imbalance'``.
        param_range : array-like
            Values to evaluate.

        Returns
        -------
        pd.DataFrame
        """
        results = []
        for value in param_range:
            params_dict = self.params.to_dict()
            params_dict[param_name] = value
            temp_params = MarketParameters(**params_dict)
            temp_opt = BalancingMarketOptimizer(temp_params, self.q)
            opt_result = temp_opt.optimize_analytical()
            results.append({
                param_name: value,
                "p_bid_optimal": opt_result.p_bid_optimal,
                "expected_revenue": opt_result.expected_revenue,
                "acceptance_prob": temp_opt.acceptance_probability(
                    opt_result.p_bid_optimal
                ),
                "beta": temp_params.beta,
            })
        return pd.DataFrame(results)
