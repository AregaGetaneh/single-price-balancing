"""
Out-of-sample evaluation framework: rolling-window backtesting and block
bootstrap confidence intervals for p^{b*}.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from .data_classes import MarketParameters
from .data_processing import estimate_parameters
from .optimization import BalancingMarketOptimizer


# ═══════════════════════════════════════════════════════════════════════
#  Rolling-window backtest
# ═══════════════════════════════════════════════════════════════════════

def rolling_backtest(
    data: pd.DataFrame,
    window_hours: int = 24 * 180,
    step_hours: int = 24 * 7,
    q: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rolling re-estimation backtest.

    At each step:
      1. Estimate parameters on the trailing window.
      2. Compute the analytical optimal bid p^{b*}.
      3. Evaluate realised revenue on the next step window.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset (time-ordered) with ``clearing_price`` and
        ``imbalance_price``.
    window_hours : int
        Training-window length (default 180 days).
    step_hours : int
        Re-estimation frequency (default 7 days).
    q : float
        Delivery position.

    Returns
    -------
    metrics : pd.DataFrame
        One row per evaluation window with columns ``train_start``,
        ``train_end``, ``test_start``, ``test_end``, ``p_bid_optimal``,
        ``mean_revenue``, ``std_revenue``, ``accept_rate``.
    realized : pd.Series
        Concatenated hourly realised revenues across all test windows.
    """
    data = data.sort_index()
    records = []
    all_revenues = []

    for t_end in range(window_hours, len(data) - step_hours, step_hours):
        train = data.iloc[t_end - window_hours : t_end]
        test = data.iloc[t_end : t_end + step_hours]

        params = estimate_parameters(train, market_name="rolling_window")
        opt = BalancingMarketOptimizer(params, q=q, data=train)
        res = opt.optimize_analytical()

        pb = res.p_bid_optimal
        rev = opt.revenue(
            pb, 1.0,
            test["clearing_price"].values,
            test["imbalance_price"].values,
        )

        records.append({
            "train_start": train.index.min(),
            "train_end": train.index.max(),
            "test_start": test.index.min(),
            "test_end": test.index.max(),
            "p_bid_optimal": pb,
            "mean_revenue": float(np.mean(rev)),
            "std_revenue": float(np.std(rev, ddof=1)),
            "accept_rate": float(np.mean(test["clearing_price"].values >= pb)),
        })
        all_revenues.append(pd.Series(rev, index=test.index))

    metrics = pd.DataFrame(records)
    realized = pd.concat(all_revenues).sort_index() if all_revenues else pd.Series(dtype=float)
    return metrics, realized


# ═══════════════════════════════════════════════════════════════════════
#  Block bootstrap for p^{b*}
# ═══════════════════════════════════════════════════════════════════════

def block_bootstrap_pb_star(
    data: pd.DataFrame,
    B: int = 500,
    block_len: int = 24 * 7,
    q: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Block bootstrap confidence interval for the Gaussian optimal bid p^{b*}.

    Resamples contiguous blocks of length ``block_len`` to preserve serial
    dependence, re-estimates the five Gaussian parameters on each bootstrap
    replicate, and computes the closed-form bid.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain ``clearing_price`` and ``imbalance_price``.
    B : int
        Number of bootstrap replicates (default 500).
    block_len : int
        Block length in hours (default 168 = 1 week).
    q : float
        Delivery position.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pb_samples : np.ndarray, shape (B,)
        Bootstrap distribution of p^{b*}.
    ci : tuple of float
        95 % confidence interval ``(lower, upper)``.
    """
    rng = np.random.default_rng(seed)
    x = data[["clearing_price", "imbalance_price"]].to_numpy()
    n = len(x)
    n_blocks = int(np.ceil(n / block_len))

    pb_samples = np.empty(B)

    for b in range(B):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        boot = np.vstack([x[s : s + block_len] for s in starts])[:n]
        boot_df = pd.DataFrame(
            boot,
            columns=["clearing_price", "imbalance_price"],
            index=data.index[:n],
        )
        params_b = estimate_parameters(boot_df, market_name="bootstrap")
        opt_b = BalancingMarketOptimizer(params_b, q=q)
        res_b = opt_b.optimize_analytical()
        pb_samples[b] = res_b.p_bid_optimal

    ci = (float(np.quantile(pb_samples, 0.025)), float(np.quantile(pb_samples, 0.975)))
    return pb_samples, ci


# ═══════════════════════════════════════════════════════════════════════
#  Simple chronological backtest (5-strategy comparison)
# ═══════════════════════════════════════════════════════════════════════

def chronological_backtest(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    q: float = 1.0,
) -> pd.DataFrame:
    """
    Simple 70/30 chronological-split backtest comparing the analytical
    Gaussian bid, empirical SAA bid, and naive benchmarks (mean, median,
    P25).

    Parameters
    ----------
    data : pd.DataFrame
    train_ratio : float
    q : float

    Returns
    -------
    pd.DataFrame
        One row per strategy with ``strategy``, ``p_bid``, ``mean_revenue``,
        ``std_revenue``, ``accept_rate``, ``sharpe``.
    """
    n_train = int(len(data) * train_ratio)
    train = data.iloc[:n_train]
    test = data.iloc[n_train:]

    lam_be_test = test["clearing_price"].values
    lam_imb_test = test["imbalance_price"].values

    params = estimate_parameters(train, market_name="train")
    opt = BalancingMarketOptimizer(params, q=q, data=train)

    # Strategies
    pb_analytical = opt.optimal_bid_price_gaussian()
    pb_empirical = opt.optimize_empirical(data=train).p_bid_optimal
    pb_mean = float(train["clearing_price"].mean())
    pb_median = float(train["clearing_price"].median())
    pb_p25 = float(train["clearing_price"].quantile(0.25))

    strategies = {
        "Analytical (Gaussian)": pb_analytical,
        "Empirical (SAA)": pb_empirical,
        "Naive (mean)": pb_mean,
        "Naive (median)": pb_median,
        "Naive (P25)": pb_p25,
    }

    rows = []
    for name, pb in strategies.items():
        rev = opt.revenue(pb, 1.0, lam_be_test, lam_imb_test)
        rows.append({
            "strategy": name,
            "p_bid": pb,
            "mean_revenue": float(np.mean(rev)),
            "std_revenue": float(np.std(rev, ddof=1)),
            "accept_rate": float(np.mean(lam_be_test >= pb)),
            "sharpe": float(np.mean(rev) / np.std(rev, ddof=1))
            if np.std(rev, ddof=1) > 0
            else 0.0,
        })

    return pd.DataFrame(rows).sort_values("mean_revenue", ascending=False)
