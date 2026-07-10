"""
regression_diagnostics.py

Reproduces the time-series diagnostics and regression estimation originally
performed in STATA (cotton_ga_2025_forecasting.do), applied here to the
national cotton acreage series:

  1. ACF / PACF plots for the key series (acreage, price ratio)
  2. Augmented Dickey-Fuller unit-root test on regression residuals
  3. OLS estimation of the acreage response model
  4. Newey-West (HAC) standard errors as a robustness check against
     residual autocorrelation

Model:
    AREA_t = b0 + b1*AREA_(t-1) + b2*PRATIO_(t-1) + b3*PRICE_COTTON_t + e_t
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from data_prep import load_panel, build_model_frame

OUT_DIR = "outputs"


def plot_acf_pacf(panel):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    plot_acf(panel["area_1000ac"], lags=10, ax=axes[0, 0], title="ACF: Cotton Acreage")
    plot_pacf(panel["area_1000ac"], lags=10, ax=axes[1, 0], title="PACF: Cotton Acreage", method="ywm")

    plot_acf(panel["pratio_cotton_corn"], lags=10, ax=axes[0, 1], title="ACF: Cotton/Corn Price Ratio")
    plot_pacf(panel["pratio_cotton_corn"], lags=10, ax=axes[1, 1], title="PACF: Cotton/Corn Price Ratio", method="ywm")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/acf_pacf_key_variables.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/acf_pacf_key_variables.png")


def fit_model(model_df):
    y = model_df["area_1000ac"]
    X = model_df[["area_lag1", "pratio_lag1", "price_cotton_cents_lb"]]
    X = sm.add_constant(X)

    ols_model = sm.OLS(y, X).fit()
    hac_model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    return ols_model, hac_model


def unit_root_test(ols_model):
    resid = ols_model.resid
    result = adfuller(resid, maxlag=2, autolag=None)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
    }


def main():
    panel = load_panel()
    model_df = build_model_frame(panel)

    plot_acf_pacf(panel)

    ols_model, hac_model = fit_model(model_df)

    print("\n=== OLS Estimation ===")
    print(ols_model.summary())

    print("\n=== Newey-West (HAC, 1 lag) Standard Errors ===")
    print(hac_model.summary())

    adf_result = unit_root_test(ols_model)
    print("\n=== ADF Test on Regression Residuals ===")
    print(f"ADF statistic: {adf_result['adf_stat']:.4f}")
    print(f"p-value:       {adf_result['p_value']:.4f}")
    print(f"Used lag:      {adf_result['used_lag']}")
    print(f"Critical values: {adf_result['critical_values']}")
    if adf_result["p_value"] < 0.05:
        print("=> Reject null of unit root: residuals appear stationary.")
    else:
        print("=> Cannot reject null of unit root at 5% level.")


if __name__ == "__main__":
    main()
