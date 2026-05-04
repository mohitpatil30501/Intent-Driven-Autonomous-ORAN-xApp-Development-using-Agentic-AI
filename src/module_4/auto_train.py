"""Two-stage auto-training: spot-check default params, then tune top performers.

Stage 1 fits every algorithm in the task-family registry once with default
params; cheap, ~seconds. Stage 2 takes the top-K performers and runs
`RandomizedSearchCV` over their hyperparameter grids, stopping as soon as one
beats the threshold. Always writes `evaluation_report.json` and saves the best
model to `saved_model.pkl`.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .registry import REGISTRY


ADMIN_COLUMNS = {"ue_id", "timestamp", "time", "index", "id"}
LABEL_NAME_CANDIDATES = ("label", "target", "y", "high_load", "anomaly", "class")


def _resolve_label_column(df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
    if hint and hint in df.columns:
        return hint
    cols_lower = {c.lower(): c for c in df.columns}
    if hint and hint.lower() in cols_lower:
        return cols_lower[hint.lower()]
    for cand in LABEL_NAME_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def _detect_task_family(cycle_type: str, label_series: Optional[pd.Series]) -> str:
    if cycle_type == "Unsupervised_ML":
        return "unsupervised_anomaly"
    if label_series is None:
        return "unsupervised_anomaly"
    if pd.api.types.is_float_dtype(label_series) and label_series.nunique(dropna=True) > 20:
        return "supervised_regression"
    n_classes = int(label_series.nunique(dropna=True))
    return "supervised_binary" if n_classes <= 2 else "supervised_multiclass"


def _step_name_for(family: str) -> str:
    if "regression" in family:
        return "regressor"
    if family.startswith("unsupervised"):
        return "detector"
    return "classifier"


def _build_pipeline(factory, family: str) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (_step_name_for(family), factory()),
    ])


def _score_supervised(y_true, y_pred, family: str) -> Tuple[str, float]:
    if family == "supervised_regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Higher is better in our framework; report as negative RMSE.
        return "neg_rmse", -rmse
    if family == "supervised_binary":
        return "f1", float(f1_score(y_true, y_pred, zero_division=0))
    return "f1_macro", float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _score_anomaly(estimator, X_test, y_test) -> Tuple[str, float]:
    raw = estimator.predict(X_test)
    pred = np.where(np.asarray(raw) == -1, 1, 0)
    if y_test is None or pd.Series(y_test).nunique(dropna=True) < 2:
        return "anomaly_separation_advisory", float(np.mean(pred))
    return "anomaly_f1", float(f1_score(y_test, pred, zero_division=0))


def _scoring_for_search(family: str) -> str:
    return {
        "supervised_binary": "f1",
        "supervised_multiclass": "f1_macro",
        "supervised_regression": "neg_root_mean_squared_error",
    }[family]


def _grid_size(param_dist: Dict[str, list]) -> int:
    n = 1
    for v in param_dist.values():
        n *= max(len(v), 1)
    return n


def run_auto_training(
    train_path: str,
    test_path: str,
    threshold: float,
    cycle_type: str,
    report_path: str,
    model_path: str,
    metric_policy: str = "task_aware",
    label_column_hint: Optional[str] = None,
    n_iter: int = 20,
    cv: int = 3,
    top_k_to_tune: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run the two-stage training pipeline. Always writes a report; returns it."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = _resolve_label_column(train_df, label_column_hint)

    excluded = set(ADMIN_COLUMNS)
    if label_col:
        excluded.add(label_col)
    feature_cols = [c for c in train_df.columns if c not in excluded]
    if not feature_cols:
        raise ValueError(
            f"No feature columns left after excluding admin/label columns. "
            f"Train columns: {list(train_df.columns)}; excluded: {sorted(excluded)}"
        )

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    if cycle_type == "Unsupervised_ML":
        y_train = None
        y_test = test_df[label_col] if label_col and label_col in test_df.columns else None
        family = "unsupervised_anomaly"
    else:
        y_train = train_df[label_col] if label_col else None
        y_test = test_df[label_col] if label_col and label_col in test_df.columns else None
        family = _detect_task_family(cycle_type, y_train)

    algorithms = REGISTRY[family]
    attempts: List[Dict[str, Any]] = []
    spot_results: List[Tuple[str, float, Pipeline]] = []

    # ===== Stage 1: spot-check =====
    for name, factory, _grid in algorithms:
        pipe = _build_pipeline(factory, family)
        try:
            if family.startswith("unsupervised"):
                fit_X = (
                    X_train[y_train == 0]
                    if y_train is not None and pd.Series(y_train).nunique(dropna=True) == 2
                    else X_train
                )
                pipe.fit(fit_X)
                metric_name, score = _score_anomaly(pipe, X_test, y_test)
            else:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                metric_name, score = _score_supervised(y_test, y_pred, family)
        except Exception as exc:
            attempts.append({
                "attempt": len(attempts) + 1,
                "stage": "spot_check",
                "model": name,
                "metric_name": "error",
                "metric_value": -1.0,
                "threshold_met": False,
                "error": str(exc)[:200],
            })
            continue

        attempts.append({
            "attempt": len(attempts) + 1,
            "stage": "spot_check",
            "model": name,
            "metric_name": metric_name,
            "metric_value": score,
            "threshold_met": score >= threshold,
        })
        spot_results.append((name, score, pipe))

    if not spot_results:
        raise RuntimeError("All spot-check attempts failed; see report for errors.")

    spot_results.sort(key=lambda t: t[1], reverse=True)
    best_name, best_score, best_model = spot_results[0]
    best_metric_name = next(
        a["metric_name"] for a in attempts
        if a["model"] == best_name and a["stage"] == "spot_check" and a["metric_name"] != "error"
    )
    best_params: Optional[Dict[str, Any]] = None

    # ===== Stage 2: tune top-K (skipped for unsupervised — no reliable CV without labels) =====
    if best_score < threshold and not family.startswith("unsupervised"):
        for name, _spot_score, _spot_pipe in spot_results[:top_k_to_tune]:
            algo = next(a for a in algorithms if a[0] == name)
            _, factory, param_dist = algo
            if not param_dist:
                continue

            pipe = _build_pipeline(factory, family)
            try:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist,
                    n_iter=min(n_iter, _grid_size(param_dist)),
                    scoring=_scoring_for_search(family),
                    cv=cv,
                    n_jobs=-1,
                    random_state=random_state,
                    refit=True,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                y_pred = search.best_estimator_.predict(X_test)
                metric_name, score = _score_supervised(y_test, y_pred, family)
            except Exception as exc:
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "stage": "tune",
                    "model": name,
                    "metric_name": "error",
                    "metric_value": -1.0,
                    "threshold_met": False,
                    "error": str(exc)[:200],
                })
                continue

            attempts.append({
                "attempt": len(attempts) + 1,
                "stage": "tune",
                "model": name,
                "metric_name": metric_name,
                "metric_value": score,
                "threshold_met": score >= threshold,
                "best_params": {k: _stringify(v) for k, v in search.best_params_.items()},
            })

            if score > best_score:
                best_score = score
                best_name = name
                best_model = search.best_estimator_
                best_metric_name = metric_name
                best_params = {k: _stringify(v) for k, v in search.best_params_.items()}

            if best_score >= threshold:
                break

    # ===== Persist artifacts =====
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(best_model, model_path)

    report = {
        "threshold": threshold,
        "metric_policy": metric_policy,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_score,
        "threshold_met": best_score >= threshold,
        "technique_used": f"{best_name} for {cycle_type}",
        "task_family": family,
        "expected_input_features": feature_cols,
        "label_column": label_col,
        "best_params": best_params,
        "evaluation_report_path": report_path,
        "attempts": attempts,
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def _stringify(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
