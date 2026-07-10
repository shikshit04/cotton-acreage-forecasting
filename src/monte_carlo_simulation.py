"""
monte_carlo_simulation.py

Reproduces the SIMETAR stochastic (Monte Carlo) forecast of 2025 national
cotton acreage. The original workbook ran 500 iterations, drawing residuals
from the empirical distribution of the regression's historical forecast
errors and propagating them through the fitted model to build a probability
distribution of plausible 2025 acreage outcomes (rather than a single
point forecast).

Approach:
  1. Fit the OLS acreage model (see regression_diagnostics.py)
  2. Bootstrap resample the historical residuals (non-parametric, so we
     don't assume normality that the data may not support)
  3. Add each resampled residual to the deterministic point forecast to
     build an empirical distribution of the 2025 acreage forecast
  4. Report mean, std dev, and a 90% simulated interval, and plot the
     resulting density
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import load_panel, build_model_frame
from regression_diagnostics import fit_model

N_ITERATIONS = 500
SEED = 42
OUT_DIR = "outputs"


def deterministic_forecast_2025(ols_model, panel):
    """
    Builds the 2025 input row: area_lag1 = 2024 acreage, pratio_lag1 = 2024
    price ratio, price_cotton = expected 2025 cotton price. Values below
    mirror the WASDE-based inputs used in the original publication
    (April 2025 WASDE: cotton-corn ratio 14.48; Feb 2025 NASS cotton price
    forecast: 67.60 cents/lb).
    """
    row_2024 = panel[panel["year"] == 2024].iloc[0]

    x_2025 = np.array([
        1.0,                      # constant
        row_2024["area_1000ac"],  # AREA_(t-1) = 2024 acreage
        14.48,                    # PRATIO_(t-1), April 2025 WASDE cotton/corn ratio
        67.60,                    # expected 2025 cotton price (cents/lb), NASS Feb 2025
    ])
    point_forecast = ols_model.params.values @ x_2025
    return point_forecast


def run_simulation(ols_model, point_forecast):
    rng = np.random.default_rng(SEED)
    historical_resid = ols_model.resid.values

    draws = rng.choice(historical_resid, size=N_ITERATIONS, replace=True)
    simulated_area = point_forecast + draws
    return simulated_area


def summarize_and_plot(simulated_area, point_forecast):
    mean_ = simulated_area.mean()
    sd_ = simulated_area.std(ddof=1)
    lower_90, upper_90 = np.percentile(simulated_area, [5, 95])

    print("=== Monte Carlo Forecast: 2025 National Cotton Acreage ===")
    print(f"Deterministic point forecast:     {point_forecast:,.1f} (1,000 acres)")
    print(f"Simulated mean (n={N_ITERATIONS}):        {mean_:,.1f}")
    print(f"Simulated std dev:                 {sd_:,.1f}")
    print(f"90% simulated interval:            [{lower_90:,.1f}, {upper_90:,.1f}]")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(simulated_area, bins=30, density=True, alpha=0.7, edgecolor="white")
    ax.axvline(mean_, color="black", linestyle="--", label=f"Simulated mean = {mean_:,.0f}")
    ax.axvline(point_forecast, color="red", linestyle=":", label=f"Deterministic forecast = {point_forecast:,.0f}")
    ax.set_title("Simulated Distribution: 2025 U.S. Cotton Planted Acreage")
    ax.set_xlabel("Planted acreage (1,000 acres)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/simulated_2025_acreage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_DIR}/simulated_2025_acreage_distribution.png")


def main():
    panel = load_panel()
    model_df = build_model_frame(panel)
    ols_model, _ = fit_model(model_df)

    point_forecast = deterministic_forecast_2025(ols_model, panel)
    simulated_area = run_simulation(ols_model, point_forecast)
    summarize_and_plot(simulated_area, point_forecast)


if __name__ == "__main__":
    main()
