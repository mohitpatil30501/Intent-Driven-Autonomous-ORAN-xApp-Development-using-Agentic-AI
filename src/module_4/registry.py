"""Per-task ML algorithm registry consumed by the auto-training runner.

Each entry is a triple `(name, factory, param_distributions)` where:
- `name` is a stable string used in evaluation reports.
- `factory` is a zero-arg callable that returns a fresh sklearn estimator with
  sensible defaults.
- `param_distributions` is the RandomizedSearchCV grid keyed by the pipeline
  step name (`classifier`, `regressor`, or `detector`). An empty dict skips
  Stage 2 tuning for that algorithm.
"""
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM


SUPERVISED_BINARY_CLASSIFIERS = [
    (
        "LogisticRegression",
        lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    (
        "RandomForestClassifier",
        lambda: RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
        {
            "classifier__n_estimators": [50, 100, 200, 400],
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
        },
    ),
    (
        "GradientBoostingClassifier",
        lambda: GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [2, 3, 5],
        },
    ),
    (
        "KNeighborsClassifier",
        lambda: KNeighborsClassifier(n_jobs=-1),
        {
            "classifier__n_neighbors": [3, 5, 7, 11, 15],
            "classifier__weights": ["uniform", "distance"],
        },
    ),
]


SUPERVISED_REGRESSORS = [
    (
        "Ridge",
        lambda: Ridge(),
        {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    ),
    (
        "RandomForestRegressor",
        lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "regressor__n_estimators": [50, 100, 200, 400],
            "regressor__max_depth": [None, 5, 10, 20],
            "regressor__min_samples_split": [2, 5, 10],
        },
    ),
    (
        "GradientBoostingRegressor",
        lambda: GradientBoostingRegressor(random_state=42),
        {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "regressor__max_depth": [2, 3, 5],
        },
    ),
]


UNSUPERVISED_ANOMALY = [
    (
        "IsolationForest",
        lambda: IsolationForest(random_state=42, contamination="auto", n_jobs=-1),
        {},
    ),
    (
        "OneClassSVM",
        lambda: OneClassSVM(),
        {},
    ),
]


# Multiclass shares the binary classifier grid; the runner picks the right scorer.
REGISTRY = {
    "supervised_binary": SUPERVISED_BINARY_CLASSIFIERS,
    "supervised_multiclass": SUPERVISED_BINARY_CLASSIFIERS,
    "supervised_regression": SUPERVISED_REGRESSORS,
    "unsupervised_anomaly": UNSUPERVISED_ANOMALY,
}
