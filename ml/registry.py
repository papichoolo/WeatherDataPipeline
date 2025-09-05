from __future__ import annotations
from typing import Literal
import mlflow
from mlflow.tracking import MlflowClient
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_NAMES

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def _pick_best_run(task: Literal['regression','classification']):
    """Return best run_id based on default metric for task."""
    if task == 'regression':
        metric = 'rmse'
        higher_is_better = False
    else:
        metric = 'f1'
        higher_is_better = True

    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError('Experiment not found')
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"], max_results=50)
    metrics_col = f"metrics.{metric}"
    if df is None or df.empty or metrics_col not in df.columns:
        raise RuntimeError('No runs with required metric found')
    row = df.iloc[0]
    return row['run_id'], float(row[metrics_col])


def promote_best(task: Literal['regression','classification']):
    run_id, score = _pick_best_run(task)
    model_name = MODEL_NAMES['REGRESSOR' if task == 'regression' else 'CLASSIFIER']

    # Find model versions created by this run
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError('No registered versions found')

    # Choose the highest version (most recent)
    version = sorted(versions, key=lambda v: int(v.version))[-1]

    # Promote to Production
    # Transition to Production (compatibility: omit archive_existing for this MLflow version)
    client.transition_model_version_stage(model_name, version.version, stage='Production')
    return {"model": model_name, "version": version.version, "stage": "Production", "score": score}
