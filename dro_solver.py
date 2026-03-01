"""
Wasserstein distributionally robust optimisation (DRO) for single-price
balancing market bidding.

The accept/reject payoff is discontinuous at p^b = λ^BE.  We smooth the
indicator with a logistic function of temperature τ and compute a local
Lipschitz proxy for the revenue gradient.  The robust lower bound is

    R̂_ε(p^b) = (1/N) Σ V_τ(Xᵢ, p^b) − ε · L̄(p^b)

where L̄ is the sample-average gradient norm (local Lipschitz constant).

References
----------
- Mohajerin Esfahani & Kuhn (2018), "Data-driven distributionally robust
  optimization using the Wasserstein metric," Math. Programming.
- Blanchet, Kang & Murthy (2019), "Robust Wasserstein profile inference
  and applications to machine learning," J. Appl. Probab.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def optimize_wasserstein_dro(
    data: pd.DataFrame,
    epsilon: float,
    q: float = 1.0,
    alpha: float = 1.0,
    tau: float = 2.0,
    pb_range: Optional[Tuple[float, float]] = None,
    norm: str = "l2",
) -> Dict:
    """
    Solve the Wasserstein DRO bidding problem for a single radius ε.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain ``clearing_price`` and ``imbalance_price``.
    epsilon : float
        Wasserstein radius (€/MWh).
    q : float
        Delivery position.
    alpha : float
        Bid quantity (1.0 = full commitment).
    tau : float
        Logistic smoothing temperature.
    pb_range : tuple, optional
        Search bounds for p^b.  Defaults to ``(μ − 3σ, μ + 2σ)`` of the
        clearing-price distribution.
    norm : ``"l2"`` or ``"l1"``
        Norm used for the Lipschitz penalty.

    Returns
    -------
    dict
        Keys: ``p_bid_optimal_dro``, ``robust_lower_bound``,
        ``empirical_mean_smoothed``, ``epsilon``, ``tau``, ``L_used``,
        ``pb_range``, ``success``, ``nfev``.
    """
    lam_be = data["clearing_price"].values.astype(float)
    lam_imb = data["imbalance_price"].values.astype(float)
    spread = lam_be - lam_imb

    if pb_range is None:
        mu = lam_be.mean()
        sd = lam_be.std(ddof=1)
        pb_range = (max(lam_be.min(), mu - 3 * sd), mu + 2 * sd)

    def robust_objective(pb: float) -> float:
        # Smoothed acceptance probability
        s = 1.0 / (1.0 + np.exp(-np.clip((lam_be - pb) / tau, -50, 50)))
        ds = (s * (1 - s)) / tau

        # Smoothed revenue
        rev = alpha * spread * s + q * lam_imb

        # Gradient w.r.t. (λ^BE, λ^imb)
        g1 = alpha * (s + spread * ds)
        g2 = q + alpha * (-s - spread * ds)

        if norm == "l2":
            grad_norm = np.sqrt(g1**2 + g2**2)
        elif norm == "l1":
            grad_norm = np.abs(g1) + np.abs(g2)
        else:
            raise ValueError("norm must be 'l2' or 'l1'")

        L_local = np.mean(grad_norm)
        robust_lb = np.mean(rev) - epsilon * L_local
        return -robust_lb

    res = minimize_scalar(robust_objective, bounds=pb_range, method="bounded")
    pb_star = float(res.x)

    # Final statistics at the optimum
    s = 1.0 / (1.0 + np.exp(-np.clip((lam_be - pb_star) / tau, -50, 50)))
    ds = (s * (1 - s)) / tau
    rev = alpha * spread * s + q * lam_imb
    g1 = alpha * (s + spread * ds)
    g2 = q + alpha * (-s - spread * ds)
    if norm == "l2":
        L_local = float(np.mean(np.sqrt(g1**2 + g2**2)))
    else:
        L_local = float(np.mean(np.abs(g1) + np.abs(g2)))

    return {
        "p_bid_optimal_dro": pb_star,
        "robust_lower_bound": float(np.mean(rev) - epsilon * L_local),
        "empirical_mean_smoothed": float(np.mean(rev)),
        "epsilon": float(epsilon),
        "tau": float(tau),
        "L_used": float(L_local),
        "pb_range": pb_range,
        "success": bool(res.success),
        "nfev": int(res.nfev),
    }


def run_dro_sensitivity(
    data: pd.DataFrame,
    epsilon_grid: List[float] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run DRO for a grid of Wasserstein radii and return a tidy DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
    epsilon_grid : list of float
        Defaults to ``[0.0, 0.25, 0.5, 1.0, 2.0]``.
    **kwargs
        Forwarded to :func:`optimize_wasserstein_dro`.

    Returns
    -------
    pd.DataFrame
        One row per ε with columns for the optimal bid, revenue bounds,
        Lipschitz constant, and conservatism cost.
    """
    if epsilon_grid is None:
        epsilon_grid = [0.0, 0.25, 0.5, 1.0, 2.0]

    rows = []
    for eps in epsilon_grid:
        res = optimize_wasserstein_dro(data, epsilon=eps, **kwargs)
        rows.append(res)

    df = pd.DataFrame(rows)

    # Conservatism cost relative to ε = 0
    base = df.loc[df["epsilon"] == 0.0, "empirical_mean_smoothed"]
    if not base.empty:
        base_val = base.iloc[0]
        df["robust_cost_pct"] = 100 * (base_val - df["robust_lower_bound"]) / base_val
    else:
        df["robust_cost_pct"] = np.nan

    return df
