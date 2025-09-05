import pandas as pd
import numpy as np

# Tiny contract
# input: DataFrame with columns: ['timestamp', 'city', 'temperature', 'humidity', 'pressure', 'wind_speed', 'weather', 'description', 'country']
# output: tuple(X_df, y_reg, y_clf) with aligned indices for supervised learning
# Notes: Assumes timestamp is Unix seconds. We sort by ['city', 'timestamp'] and create per-city lags/rollings.

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    return df


def engineer_features(df: pd.DataFrame,
                      lags: list[int] | None = None,
                      rolling_windows: list[int] | None = None,
                      inference: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    lags = lags or [1, 3]
    rolling_windows = rolling_windows or [3]

    df = df.copy()
    df = _ensure_datetime(df)

    # Sort for time-aware ops
    df.sort_values(['city', 'timestamp'], inplace=True)

    # Basic temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Create lags per city
    by_city = df.groupby('city', group_keys=False)
    for lag in lags:
        df[f'temp_lag_{lag}'] = by_city['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = by_city['humidity'].shift(lag)
        df[f'wind_lag_{lag}'] = by_city['wind_speed'].shift(lag)
        df[f'pressure_lag_{lag}'] = by_city['pressure'].shift(lag)

    # Rolling stats per city
    for w in rolling_windows:
        if w and w > 1:
            df[f'temp_rollmean_{w}'] = by_city['temperature'].rolling(w).mean().reset_index(level=0, drop=True)
            df[f'temp_rollstd_{w}'] = by_city['temperature'].rolling(w).std().reset_index(level=0, drop=True)
            df[f'humidity_rollmean_{w}'] = by_city['humidity'].rolling(w).mean().reset_index(level=0, drop=True)

    # Targets
    # Regression target: predict next-step temperature (t+1)
    df['target_temp_next'] = by_city['temperature'].shift(-1)
    # Classification target: weather condition at current timestamp
    df['target_condition'] = df['weather'].astype('category')

    # Encode categoricals for features only (city, country, weather optional)
    X = df.copy()
    y_reg = X['target_temp_next']
    y_clf = X['target_condition']

    # Feature columns
    drop_cols = {
        'description', 'target_temp_next', 'target_condition',
        'weather', 'timestamp', 'inserted_at', 'updated_at', 'batch_id', 'batch_info', 'is_current'
    }

    # One-hot encode city and hour/dayofweek; leave weather as target only
    feature_df = X.drop(columns=list(drop_cols), errors='ignore')
    feature_df = pd.get_dummies(feature_df, columns=['city', 'country', 'hour', 'dayofweek'], drop_first=True, dtype=bool)

    # Remove rows with NA introduced by lags/rolling
    if inference:
        # Keep all rows; downstream will handle NaNs appropriately
        valid_mask = pd.Series(True, index=feature_df.index)
    else:
        valid_mask = (~feature_df.isna().any(axis=1)) & (~y_reg.isna()) & (~y_clf.isna())
    feature_df = feature_df.loc[valid_mask]
    y_reg = y_reg.loc[valid_mask]
    y_clf = y_clf.loc[valid_mask]

    return feature_df, y_reg, y_clf
