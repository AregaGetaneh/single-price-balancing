# Optimal Bidding under Single-Price Imbalance Settlement

Replication code and data for:

> **Optimal bidding under single-price imbalance settlement in electricity markets: Analytical solutions and distributionally robust strategies**

## Overview

This repository implements a framework for optimal bidding in electricity balancing markets under single-price imbalance settlement (EU Regulation 2017/2195). The framework includes:

1. **Analytical closed-form bid** (Theorem 4): Under bivariate Gaussian prices, the optimal bid price satisfies the fixed-point condition $p^{b*} = E[\lambda^{imb} \mid \lambda^{BE} = p^{b*}]$, yielding

$$p^{b*} = \frac{\mu_2 - \beta \mu_1}{1 - \beta}, \qquad \beta = \rho \frac{\sigma_2}{\sigma_1}$$

2. **Empirical sample-average approximation (SAA)**: Nonparametric benchmark maximising the in-sample mean revenue.

3. **Wasserstein distributionally robust optimisation (DRO)**: Smoothed Moreau-envelope approach accommodating the discontinuous accept/reject payoff.

4. **Out-of-sample validation**: Rolling-window backtesting and block bootstrap confidence intervals on Nordic mFRR (DK2) and German aFRR (DE-LU) data (2022–2024).

## Repository Structure

```
├── config/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── __init__.py              # Package API
│   ├── data_classes.py          # MarketParameters, OptimizationResult
│   ├── data_acquisition.py      # Nordic (Energi Data Service) & German (ENTSO-E) fetchers
│   ├── data_processing.py       # Cleaning, feature engineering, parameter estimation
│   ├── optimization.py          # Analytical and numerical bid optimization
│   ├── dro_solver.py            # Wasserstein DRO with local Lipschitz proxy
│   ├── backtesting.py           # Rolling backtest, block bootstrap, strategy comparison
│   └── visualization.py         # Publication-quality figures
├── notebooks/
│   └── main_analysis.ipynb      # Full replication notebook
├── results/
│   ├── figures/                 # Generated figures (PDF + PNG)
│   └── tables/                  # Generated LaTeX tables
├── data/                        # Cached datasets (not tracked by git)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch data

```bash
# Nordic mFRR (DK2) — no API key required
python -m src.data_acquisition --market nordic

# German aFRR (DE-LU) — requires ENTSO-E API token
python -m src.data_acquisition --market german --entsoe-key YOUR_TOKEN

# Or generate synthetic data calibrated to Table 3
python -m src.data_acquisition --market nordic --synthetic
```

### 3. Run the analysis

Open and execute `notebooks/main_analysis.ipynb`, or use the library directly:

```python
from src import MarketParameters, BalancingMarketOptimizer
from src.data_acquisition import load_balancing_data
from src.data_processing import estimate_parameters

# Load data
data = load_balancing_data("nordic")

# Estimate parameters
params = estimate_parameters(data, "Nordic mFRR (DK2)")
print(params)
# MarketParameters(Nordic mFRR (DK2)):
#   μ₁ (λ^BE):   47.10 €/MWh  (σ₁=18.20)
#   μ₂ (λ^imb):  50.40 €/MWh  (σ₂=19.80)
#   ρ = 0.798,  β = 0.492

# Optimise
opt = BalancingMarketOptimizer(params, q=1.0, data=data)
result = opt.optimize_analytical()
print(result)
# OptimizationResult (analytical_gaussian):
#   p^{b*} = -156.10 €/MWh
#   α*     = 1.00
#   E[V]   = 50.42 €/MWh
```

## Data Sources

| Market | Product | Zone | Platform | Source | API |
|--------|---------|------|----------|--------|-----|
| Nordic mFRR | Manual frequency restoration reserve | DK2 | MARI | [Energi Data Service](https://www.energidataservice.dk/) | Open (no key) |
| German aFRR | Automatic frequency restoration reserve | DE-LU | PICASSO | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Requires [API token](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html) |

**Sample period**: January 2022 – February 2024 (hourly resolution, upward-activation hours only).

## Key Results

The two markets illustrate complementary regimes predicted by the theory:

| Market | β | p^{b*} (€/MWh) | 95% CI | OOS improvement |
|--------|---|-----------------|--------|-----------------|
| Nordic mFRR (DK2) | 0.49 | −156.1 | [−255, −91] | 1.6–3.1% vs naive |
| German aFRR (DE-LU) | 0.83 | 87.5 | [61, 113] | ~0% (flat surface) |

When β is well below unity, bid placement matters; when β → 1, the revenue surface flattens and all reasonable strategies perform similarly. The parameter β governs both the economic value of optimisation and the statistical precision of the optimal bid estimate.

## Citation

```bibtex
@article{singleprice2025,
  title   = {Optimal bidding under single-price imbalance settlement in
             electricity markets: Analytical solutions and distributionally
             robust strategies},
  year    = {2025},
  journal = {European Journal of Operational Research},
}
```

## License

MIT License. See [LICENSE](LICENSE).
