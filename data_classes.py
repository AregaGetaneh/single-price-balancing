"""
Core data structures for the balancing market optimization framework.

Maps code variables to paper notation:
    mu_clearing     = μ₁   (mean of λ^BE)
    sigma_clearing  = σ₁   (std of λ^BE)
    mu_imbalance    = μ₂   (mean of λ^imb)
    sigma_imbalance = σ₂   (std of λ^imb)
    rho             = ρ    (Pearson correlation)
    beta            = β    = ρ·σ₂/σ₁  (regression slope)
    p_bid_optimal   = p^{b*}  (optimal bid price)
    alpha_optimal   = α*      (optimal quantity ∈ {0, 1})
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MarketParameters:
    """
    Estimated bivariate distribution parameters for one market or estimation window.

    All price quantities are in €/MWh.
    """

    mu_clearing: float
    sigma_clearing: float
    mu_imbalance: float
    sigma_imbalance: float
    rho: float
    n_observations: int
    market_name: str

    # ── Derived quantities ──────────────────────────────────────────────

    @property
    def beta(self) -> float:
        """Regression slope: β = ρ·σ₂/σ₁."""
        if self.sigma_clearing == 0:
            return 0.0
        return self.rho * self.sigma_imbalance / self.sigma_clearing

    @property
    def mean_vector(self) -> np.ndarray:
        """μ = (μ₁, μ₂)."""
        return np.array([self.mu_clearing, self.mu_imbalance])

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Σ — 2×2 covariance matrix."""
        s1, s2, r = self.sigma_clearing, self.sigma_imbalance, self.rho
        return np.array([
            [s1**2,       r * s1 * s2],
            [r * s1 * s2, s2**2      ],
        ])

    @property
    def conditional_imbalance_std(self) -> float:
        """σ(λ^imb | λ^BE) = σ₂√(1 − ρ²)."""
        return self.sigma_imbalance * np.sqrt(max(0, 1 - self.rho**2))

    @property
    def mean_spread(self) -> float:
        """E[λ^BE − λ^imb] = μ₁ − μ₂."""
        return self.mu_clearing - self.mu_imbalance

    def conditional_imbalance_mean(self, pb: float) -> float:
        """E[λ^imb | λ^BE = p^b] = μ₂ + β·(p^b − μ₁)."""
        return self.mu_imbalance + self.beta * (pb - self.mu_clearing)

    def to_dict(self) -> dict:
        return {
            "mu_clearing": self.mu_clearing,
            "sigma_clearing": self.sigma_clearing,
            "mu_imbalance": self.mu_imbalance,
            "sigma_imbalance": self.sigma_imbalance,
            "rho": self.rho,
            "n_observations": self.n_observations,
            "market_name": self.market_name,
        }

    def __repr__(self) -> str:
        return (
            f"MarketParameters({self.market_name}):\n"
            f"  μ₁ (λ^BE):   {self.mu_clearing:.2f} €/MWh  (σ₁={self.sigma_clearing:.2f})\n"
            f"  μ₂ (λ^imb):  {self.mu_imbalance:.2f} €/MWh  (σ₂={self.sigma_imbalance:.2f})\n"
            f"  ρ = {self.rho:.3f},  β = {self.beta:.3f}\n"
            f"  E[spread] = {self.mean_spread:.2f} €/MWh\n"
            f"  σ(λ^imb|λ^BE) = {self.conditional_imbalance_std:.2f} €/MWh\n"
            f"  N = {self.n_observations:,}"
        )


@dataclass
class OptimizationResult:
    """
    Result of a single bid optimization.

    Attributes
    ----------
    p_bid_optimal : float
        Optimal bid price p^{b*} (€/MWh).
    alpha_optimal : float
        Optimal offered quantity α* ∈ {0, 1}.
    expected_revenue : float
        Expected per-unit revenue E[V] at the optimum (€/MWh).
    method : str
        Optimization method used (``"analytical_gaussian"``, ``"empirical_SAA"``,
        ``"dro_wasserstein"``).
    success : bool
        Whether the optimality condition is satisfied.
    computation_time : float, optional
        Wall-clock time in seconds.
    wasserstein_radius : float, optional
        Wasserstein radius (DRO only).
    dual_variable : float, optional
        Dual variable λ (DRO only).
    """

    p_bid_optimal: float
    alpha_optimal: float
    expected_revenue: float
    method: str
    success: bool
    computation_time: Optional[float] = None
    wasserstein_radius: Optional[float] = None
    dual_variable: Optional[float] = None

    def __repr__(self) -> str:
        lines = [
            f"OptimizationResult ({self.method}):",
            f"  p^{{b*}} = {self.p_bid_optimal:.2f} €/MWh",
            f"  α*     = {self.alpha_optimal:.2f}",
            f"  E[V]   = {self.expected_revenue:.2f} €/MWh",
        ]
        if self.wasserstein_radius is not None:
            lines.append(f"  ε      = {self.wasserstein_radius:.4f}")
        if self.computation_time is not None:
            lines.append(f"  Time   = {self.computation_time:.4f} s")
        return "\n".join(lines)
