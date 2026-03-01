"""
Publication-quality visualisation for the balancing market paper.

Generates figures referenced in the LaTeX manuscript:
    - data_visualization_pub_v2_{market}.pdf
    - conditional_linearity_{market}.pdf
    - gaussianity_check_{market}.pdf
    - backtest_results_{market}.pdf
    - sensitivity.pdf
    - bootstrap_pb_star_both.pdf
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .data_classes import MarketParameters, OptimizationResult
from .optimization import BalancingMarketOptimizer

# ── Global style ────────────────────────────────────────────────────────
STYLE = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_style():
    """Apply the publication-quality Matplotlib style."""
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update(STYLE)


def _save(fig, path: Path, pdf: bool = True):
    fig.savefig(path, bbox_inches="tight")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Conditional-linearity diagnostic ────────────────────────────────────

def plot_conditional_linearity(
    data: pd.DataFrame,
    params: MarketParameters,
    save_path: Optional[Path] = None,
    n_bins: int = 30,
):
    """
    Plot the empirical conditional mean E[λ^imb | λ^BE = x] against the
    Gaussian-implied linear conditional mean, with the 45° line showing
    the fixed-point condition.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    x = data["clearing_price"].values
    y = data["imbalance_price"].values

    # Bin and compute conditional mean / std
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    cond_means = np.full(n_bins, np.nan)
    cond_stds = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 5:
            cond_means[i] = np.mean(y[mask])
            cond_stds[i] = np.std(y[mask], ddof=1)

    valid = ~np.isnan(cond_means)

    # Gaussian-implied conditional mean
    x_grid = np.linspace(bins[0], bins[-1], 200)
    cond_gaussian = params.mu_imbalance + params.beta * (x_grid - params.mu_clearing)

    # 45° line
    diag = np.linspace(min(bins[0], cond_gaussian.min()), max(bins[-1], cond_gaussian.max()), 200)

    ax.fill_between(
        bin_centres[valid],
        cond_means[valid] - cond_stds[valid],
        cond_means[valid] + cond_stds[valid],
        alpha=0.2, color="gray", label=r"$\pm 1$ cond. std",
    )
    ax.scatter(bin_centres[valid], cond_means[valid], color="black", s=30,
               zorder=3, label="Empirical cond. mean")
    ax.plot(x_grid, cond_gaussian, "b--", lw=2,
            label=rf"Gaussian: $\mu_2 + \beta(x - \mu_1)$, $\beta={params.beta:.3f}$")
    ax.plot(diag, diag, "k:", alpha=0.5, lw=1, label=r"$45°$ line ($p^{b*}$ fixed point)")

    # Mark the fixed point
    pb_star = (params.mu_imbalance - params.beta * params.mu_clearing) / (1 - params.beta)
    ax.plot(pb_star, pb_star, "ro", ms=8, zorder=4, label=rf"$p^{{b*}} = {pb_star:.1f}$")

    ax.set_xlabel(r"$\lambda^{BE}$ (€/MWh)")
    ax.set_ylabel(r"$E[\lambda^{imb} \mid \lambda^{BE} = x]$ (€/MWh)")
    ax.set_title(f"Conditional mean diagnostic — {params.market_name}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig


# ── Gaussianity check ──────────────────────────────────────────────────

def plot_gaussianity_check(
    data: pd.DataFrame,
    params: MarketParameters,
    save_path: Optional[Path] = None,
):
    """
    Three-panel figure: histograms + Gaussian overlay for λ^BE and λ^imb,
    plus Q-Q plot for the spread.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, label, mu, sigma in [
        (axes[0], "clearing_price", r"$\lambda^{BE}$",
         params.mu_clearing, params.sigma_clearing),
        (axes[1], "imbalance_price", r"$\lambda^{imb}$",
         params.mu_imbalance, params.sigma_imbalance),
    ]:
        s = data[col].dropna()
        ax.hist(s, bins=60, density=True, alpha=0.6, edgecolor="black", lw=0.3)
        x_grid = np.linspace(s.quantile(0.005), s.quantile(0.995), 300)
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "r-", lw=2, label="Gaussian fit")
        ax.set_xlabel(f"{label} (€/MWh)")
        ax.set_ylabel("Density")
        skew, kurt = float(s.skew()), float(s.kurtosis())
        ax.set_title(f"{label}  (skew={skew:.2f}, kurt={kurt:.2f})")
        ax.legend(fontsize=8)

    # Q-Q plot for spread
    spread = (data["clearing_price"] - data["imbalance_price"]).dropna()
    stats.probplot(spread, dist="norm", plot=axes[2])
    axes[2].set_title(f"Q-Q: spread (N={len(spread):,})")
    axes[2].get_lines()[0].set_markersize(2)

    fig.suptitle(f"Gaussianity diagnostics — {params.market_name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


# ── Data overview ──────────────────────────────────────────────────────

def plot_data_overview(
    data: pd.DataFrame,
    params: MarketParameters,
    save_path: Optional[Path] = None,
):
    """
    Multi-panel overview: time series (with 7-day rolling mean), marginal
    distributions, hexbin dependence, spread distribution, Q-Q plot.
    """
    apply_style()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Row 1: time series
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(data.index, data["clearing_price"], alpha=0.3, lw=0.3, label=r"$\lambda^{BE}$")
    ax_ts.plot(data.index, data["imbalance_price"], alpha=0.3, lw=0.3, label=r"$\lambda^{imb}$")
    ax_ts.plot(data["clearing_price"].rolling(168).mean(), "b-", lw=1, label="7-day mean")
    ax_ts.plot(data["imbalance_price"].rolling(168).mean(), "r-", lw=1)
    ax_ts.set_ylabel("€/MWh")
    ax_ts.set_title(f"{params.market_name} — hourly prices")
    ax_ts.legend(fontsize=8)

    # Row 2: marginal distributions
    for i, (col, label) in enumerate([
        ("clearing_price", r"$\lambda^{BE}$"),
        ("imbalance_price", r"$\lambda^{imb}$"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        s = data[col].dropna()
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        ax.hist(s[(s >= lo) & (s <= hi)], bins=60, density=True, alpha=0.6, edgecolor="black", lw=0.3)
        ax.set_xlabel(f"{label} (€/MWh)")
        ax.set_title(f"{label} (1–99% range)")

    # Hexbin
    ax = fig.add_subplot(gs[1, 2])
    ax.hexbin(data["clearing_price"], data["imbalance_price"],
              gridsize=40, cmap="Blues", mincnt=1)
    ax.plot([data["clearing_price"].min(), data["clearing_price"].max()],
            [data["clearing_price"].min(), data["clearing_price"].max()],
            "k--", alpha=0.5, lw=1)
    ax.set_xlabel(r"$\lambda^{BE}$")
    ax.set_ylabel(r"$\lambda^{imb}$")
    ax.set_title(rf"Dependence ($\rho={params.rho:.3f}$)")

    # Row 3: spread distribution + Q-Q
    spread = data["clearing_price"] - data["imbalance_price"]
    ax = fig.add_subplot(gs[2, 0:2])
    lo, hi = spread.quantile(0.01), spread.quantile(0.99)
    ax.hist(spread[(spread >= lo) & (spread <= hi)], bins=60, density=True,
            alpha=0.6, edgecolor="black", lw=0.3)
    ax.axvline(spread.mean(), color="red", ls="--", label=f"mean={spread.mean():.1f}")
    ax.set_xlabel(r"$\lambda^{BE} - \lambda^{imb}$ (€/MWh)")
    ax.set_title("Spread distribution (1–99% range)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2])
    stats.probplot(spread.dropna(), dist="norm", plot=ax)
    ax.set_title("Q-Q: spread vs Gaussian")
    ax.get_lines()[0].set_markersize(2)

    if save_path:
        _save(fig, save_path)
    return fig


# ── Backtest comparison ────────────────────────────────────────────────

def plot_backtest_comparison(
    backtest_df: pd.DataFrame,
    test_data: pd.DataFrame,
    optimizer: BalancingMarketOptimizer,
    market_name: str = "",
    save_path: Optional[Path] = None,
):
    """
    Six-panel backtest figure: bar charts (mean revenue, Sharpe),
    acceptance–revenue scatter, cumulative revenue, and revenue
    distributions.
    """
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    df = backtest_df.sort_values("mean_revenue", ascending=False)

    # (0,0) Mean revenue bar
    ax = axes[0, 0]
    colors = ["#2196F3" if "Analytical" in s or "Empirical" in s else "#90A4AE"
              for s in df["strategy"]]
    ax.barh(df["strategy"], df["mean_revenue"], color=colors, edgecolor="black", lw=0.5)
    ax.set_xlabel("Mean revenue (€/MWh)")
    ax.set_title("Out-of-sample mean revenue")
    ax.invert_yaxis()

    # (0,1) Sharpe ratio bar
    ax = axes[0, 1]
    ax.barh(df["strategy"], df["sharpe"], color=colors, edgecolor="black", lw=0.5)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("Risk-adjusted performance")
    ax.invert_yaxis()

    # (0,2) Acceptance–revenue scatter
    ax = axes[0, 2]
    ax.scatter(df["accept_rate"] * 100, df["mean_revenue"], s=120,
               c=colors, edgecolors="black", lw=1, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["strategy"], (row["accept_rate"] * 100, row["mean_revenue"]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Acceptance rate (%)")
    ax.set_ylabel("Mean revenue (€/MWh)")
    ax.set_title("Acceptance vs revenue")

    # (1,0) Cumulative revenue
    ax = axes[1, 0]
    lam_be_test = test_data["clearing_price"].values
    lam_imb_test = test_data["imbalance_price"].values
    for _, row in df.iterrows():
        rev = optimizer.revenue(row["p_bid"], 1.0, lam_be_test, lam_imb_test)
        ax.plot(np.cumsum(rev), lw=1.2, label=row["strategy"], alpha=0.8)
    ax.set_xlabel("Test hour")
    ax.set_ylabel("Cumulative revenue (€/MWh)")
    ax.set_title("Cumulative revenue")
    ax.legend(fontsize=6, loc="upper left")

    # (1,1) Revenue distributions (top 3)
    ax = axes[1, 1]
    for _, row in df.head(3).iterrows():
        rev = optimizer.revenue(row["p_bid"], 1.0, lam_be_test, lam_imb_test)
        ax.hist(rev, bins=50, density=True, alpha=0.5, label=row["strategy"])
    ax.set_xlabel("Hourly revenue (€/MWh)")
    ax.set_title("Revenue distribution (top 3)")
    ax.legend(fontsize=7)

    # (1,2) Summary text
    ax = axes[1, 2]
    ax.axis("off")
    best = df.iloc[0]
    worst = df.iloc[-1]
    txt = (
        f"Market: {market_name}\n"
        f"Test hours: {len(test_data):,}\n\n"
        f"Best strategy: {best['strategy']}\n"
        f"  p^b = {best['p_bid']:.1f} €/MWh\n"
        f"  E[V] = {best['mean_revenue']:.2f} €/MWh\n"
        f"  Accept = {best['accept_rate']:.1%}\n\n"
        f"Spread vs worst: {best['mean_revenue'] - worst['mean_revenue']:.2f} €/MWh "
        f"({100*(best['mean_revenue']/worst['mean_revenue']-1):.1f}%)"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Out-of-sample backtest — {market_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig
