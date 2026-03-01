"""
Data processing utilities: cleaning, feature engineering, parameter estimation,
normality testing, and train/test splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats

from .data_classes import MarketParameters


# ── Parameter estimation ────────────────────────────────────────────────

def estimate_parameters(df: pd.DataFrame, market_name: str) -> MarketParameters:
    """
    Estimate bivariate distribution parameters from a DataFrame
    containing ``clearing_price`` and ``imbalance_price`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``clearing_price`` and ``imbalance_price``.
    market_name : str
        Label for the market (used in printing/logging).

    Returns
    -------
    MarketParameters
    """
    return MarketParameters(
        mu_clearing=float(df["clearing_price"].mean()),
        sigma_clearing=float(df["clearing_price"].std(ddof=1)),
        mu_imbalance=float(df["imbalance_price"].mean()),
        sigma_imbalance=float(df["imbalance_price"].std(ddof=1)),
        rho=float(df["clearing_price"].corr(df["imbalance_price"])),
        n_observations=len(df),
        market_name=market_name,
    )


# ── Feature engineering ─────────────────────────────────────────────────

def prepare_market_data(df: pd.DataFrame, market_name: str = "") -> pd.DataFrame:
    """
    Add temporal features and the spread to raw market data.

    Parameters
    ----------
    df : pd.DataFrame
        DatetimeIndex + columns ``[clearing_price, imbalance_price]``.
    market_name : str
        Label for logging.

    Returns
    -------
    pd.DataFrame
        Input data augmented with ``hour``, ``day_of_week``, ``month``,
        ``year``, ``quarter``, ``is_weekend``, ``is_peak``, ``spread``,
        ``abs_spread``.
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_peak"] = (
        (df["hour"] >= 8)
        & (df["hour"] < 20)
        & (~df["is_weekend"].astype(bool))
    ).astype(int)
    df["spread"] = df["clearing_price"] - df["imbalance_price"]
    df["abs_spread"] = df["spread"].abs()
    return df


# ── Cleaning ────────────────────────────────────────────────────────────

def clean_data(
    df: pd.DataFrame,
    price_floor: float = -500.0,
    price_ceiling: float = 4000.0,
) -> pd.DataFrame:
    """
    Remove rows with missing or extreme prices.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data.
    price_floor, price_ceiling : float
        Observations with any price outside ``[floor, ceiling]`` are dropped.

    Returns
    -------
    pd.DataFrame
    """
    df = df.dropna(subset=["clearing_price", "imbalance_price"])
    mask = (
        (df["clearing_price"] >= price_floor)
        & (df["clearing_price"] <= price_ceiling)
        & (df["imbalance_price"] >= price_floor)
        & (df["imbalance_price"] <= price_ceiling)
    )
    return df.loc[mask].copy()


# ── Normality testing ───────────────────────────────────────────────────

def test_normality(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Run Jarque–Bera and Shapiro–Wilk tests on clearing price, imbalance
    price, and the spread.

    Returns
    -------
    dict
        Nested dict keyed by series name → {statistic, p_value, reject_5pct}.
    """
    results = {}
    series_map = {
        "clearing_price": df["clearing_price"],
        "imbalance_price": df["imbalance_price"],
        "spread": df["clearing_price"] - df["imbalance_price"],
    }
    for name, s in series_map.items():
        jb_stat, jb_p = stats.jarque_bera(s.dropna())
        results[name] = {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "skewness": float(s.skew()),
            "excess_kurtosis": float(s.kurtosis()),
            "reject_5pct": jb_p < 0.05,
        }
    return results


# ── Summary statistics ──────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a summary-statistics table for both price series and the spread.
    """
    spread = df["clearing_price"] - df["imbalance_price"]
    records = []
    for name, s in [
        ("clearing_price", df["clearing_price"]),
        ("imbalance_price", df["imbalance_price"]),
        ("spread", spread),
    ]:
        records.append({
            "series": name,
            "mean": s.mean(),
            "std": s.std(ddof=1),
            "skewness": s.skew(),
            "kurtosis": s.kurtosis(),
            "min": s.min(),
            "p5": s.quantile(0.05),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
            "max": s.max(),
            "N": len(s),
        })
    return pd.DataFrame(records).set_index("series")


# ── Train / test split ──────────────────────────────────────────────────

def split_train_test(
    df: pd.DataFrame, train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split (no shuffling).
    """
    n_train = int(len(df) * train_ratio)
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def subsample_data(
    df: pd.DataFrame,
    start: str = None,
    end: str = None,
    year: int = None,
    n_hours: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Subsample real data by date range, year, or random draw.
    """
    if year is not None:
        return df.loc[str(year)]
    if start is not None:
        return df.loc[start:end]
    if n_hours is not None and n_hours < len(df):
        return df.sample(n=n_hours, random_state=seed).sort_index()
    return df
