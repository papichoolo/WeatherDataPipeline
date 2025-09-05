from __future__ import annotations
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, MODEL_NAMES
from .features import engineer_features, _ensure_datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import json
import os
import tempfile

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def _load_production_model(model_name: str):
    """Load a model from the registry.

    Prefer the sklearn flavor to avoid strict MLflow input schema enforcement during
    inference. Fall back to the generic pyfunc flavor if needed.
    """
    # Load from Registry stage 'Production' if available; otherwise latest version
    try:
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        # Fallback to latest version
        latest = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"]) or []
        if not latest:
            raise RuntimeError(f"No versions found for model {model_name}")
        # pick highest version
        version = sorted(latest, key=lambda m: int(m.version))[-1]
        model_uri = f"models:/{model_name}/{version.version}"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            return mlflow.pyfunc.load_model(model_uri)


def _get_feature_columns(model_name: str):
    # Try to get feature_columns.json from the model's run artifacts
    versions = client.get_latest_versions(model_name, stages=["Production"]) or []
    if not versions:
        versions = client.get_latest_versions(model_name, stages=["Staging", "None"]) or []
    if not versions:
        raise RuntimeError(f"No versions found for {model_name}")
    mv = sorted(versions, key=lambda m: int(m.version))[-1]
    run_id = mv.run_id
    with tempfile.TemporaryDirectory() as td:
        try:
            local = client.download_artifacts(run_id, "feature_columns.json", td)
            with open(local, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("feature_columns", [])
        except Exception:
            return []


def _align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return X
    X = X.copy()
    # Add missing columns with zeros
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        if c.startswith(('city_', 'country_', 'hour_', 'dayofweek_')):
            X[c] = False
        else:
            X[c] = 0.0
    # Keep only known columns and order them
    X = X[feature_columns]
    # Fill remaining NaN
    for col in X.columns:
        if col.startswith(('city_', 'country_', 'hour_', 'dayofweek_')):
            X[col] = X[col].astype('bool').fillna(False)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64').fillna(0.0)
    # Align known integer fields expected by training schema
    for int_col in ('humidity', 'pressure'):
        if int_col in X.columns:
            X[int_col] = pd.to_numeric(X[int_col], errors='coerce').fillna(0).astype('int64')
    return X


def _prep_features_for_inference(df: pd.DataFrame, model_name_for_schema: str) -> pd.DataFrame:
    X, _, _ = engineer_features(df, inference=True)
    cols = _get_feature_columns(model_name_for_schema)
    X = _align_features(X, cols)
    return X


def predict(df: pd.DataFrame):
    # Ensure types and create features
    df = _ensure_datetime(df)
    # Use regressor schema for both to keep consistent features
    X = _prep_features_for_inference(df, MODEL_NAMES["REGRESSOR"])

    reg_model = _load_production_model(MODEL_NAMES["REGRESSOR"])
    # Realign in case classifier expects slightly different schema
    X_clf = _prep_features_for_inference(df, MODEL_NAMES["CLASSIFIER"])
    clf_model = _load_production_model(MODEL_NAMES["CLASSIFIER"])

    temp_pred = reg_model.predict(X)
    cond_pred = clf_model.predict(X_clf)

    return pd.DataFrame({
        "pred_temperature": np.asarray(temp_pred).ravel(),
        "pred_condition": np.asarray(cond_pred).ravel()
    }, index=X.index)


def evaluate_and_log(df: pd.DataFrame):
    """Compute metrics using current Production models on provided data and log to MLflow."""
    df = _ensure_datetime(df)
    # Build training-style features with fallbacks to avoid empty sets on small windows
    def _build_eval_features_with_fallback(_df: pd.DataFrame):
        X0, yr0, yc0 = engineer_features(_df, inference=False)
        if len(X0) > 0:
            return X0, yr0, yc0
        X1, yr1, yc1 = engineer_features(_df, lags=[1], rolling_windows=[2], inference=False)
        if len(X1) > 0:
            return X1, yr1, yc1
        X2, yr2, yc2 = engineer_features(_df, lags=[1], rolling_windows=[], inference=False)
        return X2, yr2, yc2

    # Build training-style features to ensure targets exist and rows are valid
    X_full, y_reg, y_clf = _build_eval_features_with_fallback(df)
    if len(X_full) == 0:
        raise RuntimeError("No valid rows for evaluation after feature engineering (even after fallbacks).")

    reg = _load_production_model(MODEL_NAMES["REGRESSOR"])
    clf = _load_production_model(MODEL_NAMES["CLASSIFIER"])

    # Align features to each model's schema
    X_reg = _align_features(X_full, _get_feature_columns(MODEL_NAMES["REGRESSOR"]))
    X_clf = _align_features(X_full, _get_feature_columns(MODEL_NAMES["CLASSIFIER"]))

    # After alignment, ensure no NaNs remain; keep rows valid for both tasks
    valid_idx = X_reg.index.intersection(X_clf.index)
    X_reg = X_reg.loc[valid_idx]
    X_clf = X_clf.loc[valid_idx]
    y_reg = y_reg.loc[valid_idx]
    y_clf = y_clf.loc[valid_idx]
    # Drop any rows with residual NaNs just in case
    mask = (~X_reg.isna().any(axis=1)) & (~X_clf.isna().any(axis=1)) & (~y_reg.isna()) & (~y_clf.isna())
    X_reg = X_reg.loc[mask]
    X_clf = X_clf.loc[mask]
    y_reg = y_reg.loc[mask]
    y_clf = y_clf.loc[mask]
    if len(X_reg) == 0:
        raise RuntimeError("No rows left after alignment and NaN filtering for evaluation.")

    yhat_reg = reg.predict(X_reg)
    # Ensure targets match dtypes for metrics
    y_reg = pd.to_numeric(y_reg, errors='coerce')
    yhat_reg = np.asarray(yhat_reg).astype(float)
    yhat_clf = clf.predict(X_clf)
    # For classification, coerce targets to string labels to match RF output
    y_clf = y_clf.astype(str)
    yhat_clf = np.asarray(yhat_clf).astype(str)

    mae = float(mean_absolute_error(y_reg, yhat_reg))
    rmse = float(np.sqrt(mean_squared_error(y_reg, yhat_reg)))
    acc = float(accuracy_score(y_clf, yhat_clf))
    f1 = float(f1_score(y_clf, yhat_clf, average="weighted"))

    try:
        with mlflow.start_run(run_name="inference-eval"):
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "accuracy": acc, "f1": f1})
    except Exception:
        pass

    return {"mae": mae, "rmse": rmse, "accuracy": acc, "f1": f1, "n": int(len(X_reg))}


def evaluate_with_details(df: pd.DataFrame):
    """Evaluate like evaluate_and_log, but also return a DataFrame of predictions vs. actuals.

    Returns: (metrics_dict, details_df)
    details_df columns include: city, timestamp, actual_temp_next, actual_condition, pred_temperature, pred_condition
    """
    df = _ensure_datetime(df)

    def _build_eval_features_with_fallback(_df: pd.DataFrame):
        X0, yr0, yc0 = engineer_features(_df, inference=False)
        if len(X0) > 0:
            return X0, yr0, yc0
        X1, yr1, yc1 = engineer_features(_df, lags=[1], rolling_windows=[2], inference=False)
        if len(X1) > 0:
            return X1, yr1, yc1
        X2, yr2, yc2 = engineer_features(_df, lags=[1], rolling_windows=[], inference=False)
        return X2, yr2, yc2

    # Keep a copy of essential raw columns aligned by index
    base_cols = [c for c in ['city', 'timestamp', 'temperature', 'weather', 'country', 'humidity', 'pressure'] if c in df.columns]
    raw = df.copy()

    X_full, y_reg, y_clf = _build_eval_features_with_fallback(df)
    if len(X_full) == 0:
        raise RuntimeError("No valid rows for evaluation after feature engineering (even after fallbacks).")

    reg = _load_production_model(MODEL_NAMES["REGRESSOR"])
    clf = _load_production_model(MODEL_NAMES["CLASSIFIER"])

    X_reg = _align_features(X_full, _get_feature_columns(MODEL_NAMES["REGRESSOR"]))
    X_clf = _align_features(X_full, _get_feature_columns(MODEL_NAMES["CLASSIFIER"]))

    valid_idx = X_reg.index.intersection(X_clf.index)
    X_reg = X_reg.loc[valid_idx]
    X_clf = X_clf.loc[valid_idx]
    y_reg = y_reg.loc[valid_idx]
    y_clf = y_clf.loc[valid_idx]

    mask = (~X_reg.isna().any(axis=1)) & (~X_clf.isna().any(axis=1)) & (~y_reg.isna()) & (~y_clf.isna())
    X_reg = X_reg.loc[mask]
    X_clf = X_clf.loc[mask]
    y_reg = pd.to_numeric(y_reg.loc[mask], errors='coerce')
    y_clf = y_clf.loc[mask].astype(str)
    idx = X_reg.index
    if len(idx) == 0:
        raise RuntimeError("No rows left after alignment and NaN filtering for evaluation.")

    yhat_reg = np.asarray(reg.predict(X_reg)).astype(float)
    yhat_clf = np.asarray(clf.predict(X_clf)).astype(str)

    mae = float(mean_absolute_error(y_reg, yhat_reg))
    rmse = float(np.sqrt(mean_squared_error(y_reg, yhat_reg)))
    acc = float(accuracy_score(y_clf, yhat_clf))
    f1 = float(f1_score(y_clf, yhat_clf, average="weighted"))

    try:
        with mlflow.start_run(run_name="inference-eval"):
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "accuracy": acc, "f1": f1})
    except Exception:
        pass

    # Build details DataFrame
    base = raw.loc[idx, base_cols].copy() if base_cols else pd.DataFrame(index=idx)
    details = base.assign(
        actual_temp_next=y_reg.values,
        actual_condition=y_clf.values,
        pred_temperature=yhat_reg,
        pred_condition=yhat_clf,
    )

    return {"mae": mae, "rmse": rmse, "accuracy": acc, "f1": f1, "n": int(len(details))}, details
