"""
Single-price balancing market optimal bidding framework.

Implements the analytical, empirical, and distributionally robust bidding
strategies described in:

    "Optimal bidding under single-price imbalance settlement in electricity
     markets: Analytical solutions and distributionally robust strategies"

Modules
-------
data_classes     : Core data structures (MarketParameters, OptimizationResult)
data_acquisition : Nordic mFRR and German aFRR data fetching
data_processing  : Cleaning, feature engineering, parameter estimation
optimization     : Analytical and numerical bid optimization
dro_solver       : Wasserstein distributionally robust optimization
backtesting      : Out-of-sample evaluation framework
visualization    : Publication-quality figure generation
"""

from .data_classes import MarketParameters, OptimizationResult
from .optimization import BalancingMarketOptimizer
from .dro_solver import optimize_wasserstein_dro
from .backtesting import rolling_backtest, block_bootstrap_pb_star

__version__ = "1.0.0"
