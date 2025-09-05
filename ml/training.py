from __future__ import annotations
import os
from typing import Literal, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_NAMES
from ml.features import engineer_features
from load import get_weather_data_from_mongodb
from config import MONGODB_PASSWORD, COLLECTIONS

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
_client = MlflowClient()


def _tscv_splits(n_splits: int = 5):
    return TimeSeriesSplit(n_splits=n_splits)


def _build_features_with_fallback(raw_df: pd.DataFrame):
    # Try default (lags=[1,3], rolling=[3])
    X, y_reg, y_clf = engineer_features(raw_df)
    if len(X) >= 50:
        return X, y_reg, y_clf
    # Fallback 1: lighter features (lags=[1], rolling=[2])
    X, y_reg, y_clf = engineer_features(raw_df, lags=[1], rolling_windows=[2])
    if len(X) >= 30:
        return X, y_reg, y_clf
    # Fallback 2: minimal temporal (lags=[1], no rolling)
    X, y_reg, y_clf = engineer_features(raw_df, lags=[1], rolling_windows=[])
    return X, y_reg, y_clf


def train_regressor(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                    params: Optional[dict] = None):
    params = params or {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
    model = RandomForestRegressor(**params)

    tscv = _tscv_splits(n_splits)
    fold_metrics = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_metrics.append({"mae": mae, "rmse": rmse})

    # Retrain on full data
    model.fit(X, y)
    avg_mae = float(np.mean([m["mae"] for m in fold_metrics]))
    avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    return model, {"mae": avg_mae, "rmse": avg_rmse}


def train_classifier(X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                     algo: Literal["rf", "logreg"] = "rf",
                     params: Optional[dict] = None):
    if algo == "rf":
        params = params or {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
        model = RandomForestClassifier(**params)
    else:
        params = params or {"max_iter": 1000}
        model = LogisticRegression(**params)

    tscv = _tscv_splits(n_splits)
    fold_metrics = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="weighted")
        fold_metrics.append({"accuracy": acc, "f1": f1})

    model.fit(X, y)
    avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    avg_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    return model, {"accuracy": avg_acc, "f1": avg_f1}


def train_and_log(raw_df: pd.DataFrame, clf_algo: Literal["rf", "logreg"] = "rf"):
    X, y_reg, y_clf = _build_features_with_fallback(raw_df)
    if len(X) < 20:
        raise RuntimeError("Not enough data to train models. Run more ETL cycles or relax feature windows.")

    # Regressor
    with mlflow.start_run(run_name="regressor") as run_reg:
        reg_model, reg_metrics = train_regressor(X, y_reg)
        mlflow.log_params({"task": "regression", "model": "RandomForestRegressor"})
        mlflow.log_metrics(reg_metrics)
        # Log feature schema
        try:
            mlflow.log_dict({"feature_columns": list(X.columns)}, artifact_file="feature_columns.json")
        except Exception:
            pass
        signature = infer_signature(X, reg_model.predict(X))
        mlflow.sklearn.log_model(reg_model, artifact_path="model",
                                 signature=signature,
                                 registered_model_name=MODEL_NAMES["REGRESSOR"])
        # Transition latest to Staging
        try:
            latest = _client.get_latest_versions(MODEL_NAMES["REGRESSOR"], stages=["None"]) or []
            if latest:
                v = sorted(latest, key=lambda m: int(m.version))[-1]
                _client.transition_model_version_stage(MODEL_NAMES["REGRESSOR"], v.version, stage="Staging", archive_existing=False)
        except Exception:
            pass

    # Classifier
    with mlflow.start_run(run_name="classifier") as run_clf:
        clf_model, clf_metrics = train_classifier(X, y_clf, algo=clf_algo)
        algo_name = "RandomForestClassifier" if clf_algo == "rf" else "LogisticRegression"
        mlflow.log_params({"task": "classification", "model": algo_name})
        mlflow.log_metrics(clf_metrics)
        # Log feature schema
        try:
            mlflow.log_dict({"feature_columns": list(X.columns)}, artifact_file="feature_columns.json")
        except Exception:
            pass
        signature = infer_signature(X, clf_model.predict(X))
        mlflow.sklearn.log_model(clf_model, artifact_path="model",
                                 signature=signature,
                                 registered_model_name=MODEL_NAMES["CLASSIFIER"])
        try:
            latest = _client.get_latest_versions(MODEL_NAMES["CLASSIFIER"], stages=["None"]) or []
            if latest:
                v = sorted(latest, key=lambda m: int(m.version))[-1]
                _client.transition_model_version_stage(MODEL_NAMES["CLASSIFIER"], v.version, stage="Staging", archive_existing=False)
        except Exception:
            pass

    return True


def retrain_from_mongo(db_password: str | None = None, collection: str | None = None):
    db_password = db_password or MONGODB_PASSWORD
    collection = collection or COLLECTIONS["RAW_DATA"]
    df = get_weather_data_from_mongodb(collection, db_password)
    if df is None or df.empty:
        raise RuntimeError("No training data found in MongoDB.")
    # Ensure numeric types
    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols + ["weather", "timestamp", "city", "country"])  # basic cleanliness
    return train_and_log(df)
