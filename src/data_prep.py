"""
data_prep.py

Loads the raw national cotton panel (1990-2024) and constructs the lagged
variables used in the forecasting model:

    AREA_t = f(AREA_t-1, PRATIO_t-1, PRICE_COTTON_t)

where PRATIO = cotton price (cents/lb) / corn price ($/bu), following the
convention used in Parajuli & Liu (2025), "Projections for 2025 Cotton
Acreage," Southern Ag Today 5(18.3).
"""

import pandas as pd

DATA_PATH = "data/cotton_national_1990_2024.csv"


def load_panel(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    return df


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Construct one-year lags needed for the acreage response model."""
    df = df.copy()
    df["area_lag1"] = df["area_1000ac"].shift(1)
    df["pratio_lag1"] = df["pratio_cotton_corn"].shift(1)
    df["pratio_lag2"] = df["pratio_cotton_corn"].shift(2)
    df["price_cotton_lag1"] = df["price_cotton_cents_lb"].shift(1)
    df["price_corn_lag1"] = df["price_corn_dollar_bu"].shift(1)

    # Model estimation sample drops the first lagged observation(s)
    model_df = df.dropna().reset_index(drop=True)
    return model_df


if __name__ == "__main__":
    panel = load_panel()
    model_df = build_model_frame(panel)
    print(panel.head())
    print("\nModel-ready frame (post-lag):")
    print(model_df.head())
    print(f"\n{len(model_df)} usable observations for estimation.")
