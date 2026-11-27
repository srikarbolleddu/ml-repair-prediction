"""Reproducible training: download the pinned dataset → clean → compare models by maintenance cost →
write artifacts.

    uv run python training/train.py

Writes:
  app/model_artifacts/model.skops, manifest.json, samples.json, metrics.json, explore.json
  reports/metrics.json      held-out metrics -- the only source for numbers quoted anywhere
  reports/explore.json      precomputed data for the demo's interactive "Explore" charts
  reports/golden.json       fixed inputs + expected outputs, checked by tests/test_golden.py
  reports/figures/*.png     charts for the README / case study / blog

Method follows the UT Austin PGP-AIML ReneWind project (predict failures from sensor readings and
choose the model by the cost of mistakes: replacement ≫ repair ≫ inspection), rebuilt on the
openly licensed AI4I 2020 dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import model as model_io  # noqa: E402
from app.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    FEATURE_ORDER,
    FIELDS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_raw,
    records_to_frame,
)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
DATA_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
CACHE = ROOT / ".data" / "ai4i2020.zip"
SEED = 42
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Illustrative maintenance costs (USD). A missed failure costs 40x a false alarm.
COSTS = {"false_alarm": 1_000, "repair": 10_000, "replacement": 40_000}
DERIVED_LABELS = {
    "power_w": "Mechanical power (derived)",
    "temp_diff_k": "Process − air temperature (derived)",
    "strain_min_nm": "Tool wear × torque (derived)",
}


def fetch_dataset() -> pd.DataFrame:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        # DATA_URL is a constant https URL and the payload is checksum-verified below.
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:  # nosec B310
            CACHE.write_bytes(response.read())
    payload = CACHE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise SystemExit(f"dataset checksum mismatch: {digest} != {DATA_SHA256}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return pd.read_csv(io.BytesIO(archive.read("ai4i2020.csv")), encoding="utf-8-sig")


def build_pipeline(estimator: Any, numeric: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def candidates() -> dict[str, list[tuple[Any, list[str]]]]:
    """Model family → hyperparameter options, each with the numeric columns it uses."""
    with_physics = [*NUMERIC_FEATURES, *DERIVED_FEATURES]
    return {
        "Logistic regression": [
            (LogisticRegression(C=c, max_iter=2000, random_state=SEED), with_physics)
            for c in (0.1, 1.0, 10.0)
        ],
        "Neural network (MLP)": [
            (
                MLPClassifier(
                    hidden_layer_sizes=layers,
                    alpha=alpha,
                    max_iter=600,
                    early_stopping=True,
                    random_state=SEED,
                ),
                with_physics,
            )
            for layers in ((32,), (64, 32))
            for alpha in (1e-4, 1e-2)
        ],
        "Gradient boosting": [
            (
                HistGradientBoostingClassifier(
                    learning_rate=lr, max_leaf_nodes=leaves, max_iter=300, random_state=SEED
                ),
                with_physics,
            )
            for lr in (0.05, 0.1)
            for leaves in (15, 31)
        ],
        "Gradient boosting, raw sensors only": [
            (
                HistGradientBoostingClassifier(
                    learning_rate=lr, max_leaf_nodes=leaves, max_iter=300, random_state=SEED
                ),
                list(NUMERIC_FEATURES),
            )
            for lr in (0.05, 0.1)
            for leaves in (15, 31)
        ],
    }


def proba(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(x)[:, 1]


def maintenance_cost(y_true: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, flagged, labels=[0, 1]).ravel())
    cost = fp * COSTS["false_alarm"] + tp * COSTS["repair"] + fn * COSTS["replacement"]
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "cost": cost}


def cost_per_1000(y_true: np.ndarray, p: np.ndarray, threshold: float) -> float:
    return maintenance_cost(y_true, p >= threshold)["cost"] / len(y_true) * 1000


def cheapest_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """The cutoff with the lowest total maintenance cost (on out-of-fold predictions)."""
    # Fine steps at the low end: when a miss costs 40x a false alarm, the best cutoff is small.
    grid = np.unique(
        np.round(np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.05, 0.99, 95)]), 3)
    )
    return float(min(grid, key=lambda t: maintenance_cost(y_true, p >= t)["cost"]))


def evaluate(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = p >= threshold
    counts = maintenance_cost(y_true, flagged)
    no_model = int(y_true.sum()) * COSTS["replacement"]  # run to failure: every failure missed
    return {
        "recall": round(float(recall_score(y_true, flagged)), 4),
        "precision": round(float(precision_score(y_true, flagged, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "false_alarms_per_1000": round(counts["fp"] / len(y_true) * 1000, 1),
        "cost_per_1000": round(counts["cost"] / len(y_true) * 1000),
        "savings_vs_no_model": round(1 - counts["cost"] / no_model, 4),
        **{k: counts[k] for k in ("tn", "fp", "fn", "tp")},
    }


def main() -> None:
    df = clean_raw(fetch_dataset())
    x, y = df[MODEL_FEATURES], df[TARGET]
    # 80/20 stratified. With only ~3% failures, a single 2,000-row validation split holds ~68
    # failures -- too few to choose between models reliably (an earlier version did, and picked
    # the weaker model). So choices are made by 5-fold cross-validation on the 80%, and the 20%
    # test set is used once at the end.
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED
    )
    yd, yt = y_dev.to_numpy(), y_test.to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"rows: development={len(x_dev)} (5-fold CV) test={len(x_test)}")

    fitted: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    cv_scores: dict[str, dict[str, float]] = {}
    for family, options in candidates().items():
        best, best_cost, best_t, best_oof = None, float("inf"), 0.5, None
        for estimator, numeric in options:
            pipeline = build_pipeline(estimator, numeric)
            # Out-of-fold probabilities: every development row is scored by a model that
            # didn't see it, so the cost and cutoff are chosen on 8,000 honest predictions.
            oof = cross_val_predict(pipeline, x_dev, y_dev, cv=folds, method="predict_proba")[:, 1]
            t = cheapest_threshold(yd, oof)
            cost = maintenance_cost(yd, oof >= t)["cost"]
            if cost < best_cost:
                best, best_cost, best_t, best_oof = pipeline, cost, t, oof
        if best is None or best_oof is None:
            raise RuntimeError(f"no model for {family}")
        fitted[family] = best.fit(x_dev, y_dev)
        thresholds[family] = best_t
        cv_scores[family] = evaluate(yd, best_oof, best_t)
        print(f"{family}: CV cost ${best_cost:,} at cutoff {best_t}")

    comparison: dict[str, dict[str, Any]] = {
        name: {
            "threshold": thresholds[name],
            "cross_validation": cv_scores[name],
            "test": evaluate(yt, proba(p, x_test), thresholds[name]),
        }
        for name, p in fitted.items()
    }
    final_name = min(fitted, key=lambda n: comparison[n]["cross_validation"]["cost_per_1000"])
    final, threshold = fitted[final_name], thresholds[final_name]
    test_p = proba(final, x_test)
    x_train = x_dev  # medians for the explanation baseline come from here

    # Interpretation only (no choice depends on it), so the untouched test set is the fairest data.
    perm: Any = permutation_importance(
        final, x_test, y_test, scoring="average_precision", n_repeats=10, random_state=SEED
    )
    labels = {f.name: f.label for f in FIELDS} | DERIVED_LABELS
    importance_rows = sorted(
        (
            {"field": n, "label": labels[n], "importance": round(float(v), 4)}
            for n, v in zip(MODEL_FEATURES, perm.importances_mean, strict=True)
        ),
        key=lambda r: -float(r["importance"]),
    )

    baseline: dict[str, Any] = {n: float(x_train[n].median()) for n in NUMERIC_FEATURES}
    baseline.update({n: str(x_train[n].mode().iloc[0]) for n in CATEGORICAL_FEATURES})

    params_hash = hashlib.sha256(str(final.get_params()).encode()).hexdigest()[:8]
    model_version = f"{final_name.split()[0].lower()}-{params_hash}"
    artifacts = model_io.ARTIFACT_DIR
    artifacts.mkdir(parents=True, exist_ok=True)
    model_io.save(final, model_io.MODEL_PATH)
    manifest = {
        "model_version": model_version,
        "model": final_name,
        "failure_threshold": threshold,
        "baseline": baseline,
        "costs": COSTS,
        "dataset": {"url": DATA_URL, "sha256": DATA_SHA256, "license": "CC BY 4.0"},
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Real held-out machines, both outcomes, chosen deterministically.
    t = x_test.assign(actual=yt, p=test_p).sort_values("p")
    failed, healthy = t[t.actual == 1], t[t.actual == 0]
    wanted = [
        ("Healthy: new tool, normal load", healthy[healthy.tool_wear_min < 30], 0.5),
        ("Healthy: worn tool, light load", healthy[healthy.tool_wear_min > 180], 0.5),
        ("False alarm: flagged, but it kept running", healthy[healthy.p >= threshold], 0.5),
        ("Failed: worn tool under high torque", failed[failed.strain_min_nm > 11_000], 1.0),
        ("Failed: overheating (small temperature gap)", failed[failed.temp_diff_k < 8.6], 1.0),
        ("Failed, but the model missed it", failed[failed.p < threshold], 1.0),
    ]
    used: set[Any] = set()
    samples: list[dict[str, Any]] = []
    for title, group, position in wanted:
        free = group.drop(index=[i for i in group.index if i in used])
        if free.empty:
            continue
        row = free.iloc[min(int(len(free) * position), len(free) - 1)]
        used.add(row.name)
        inputs: dict[str, Any] = {}
        for name in FEATURE_ORDER:
            value = row[name]
            if name in NUMERIC_FEATURES:
                value = round(float(value), 1)
                value = int(value) if value.is_integer() else value
            inputs[name] = value
        samples.append(
            {"title": title, "inputs": inputs, "actual": "failed" if row.actual else "kept running"}
        )
    (artifacts / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    REPORTS.mkdir(exist_ok=True)
    metrics = {
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance (synthetic)",
            "rows": len(df),
            "positive_rate": round(float(y.mean()), 4),
            "positive_class": "failure",
            "split": {"development_5fold_cv": len(x_dev), "test": len(x_test)},
            "excluded_columns": ["TWF, HDF, PWF, OSF, RNF (failure-mode flags: target leakage)"],
        },
        "costs": COSTS,
        "final_model": final_name,
        "threshold": threshold,
        "test": comparison[final_name]["test"],
        "comparison": comparison,
        "model_version": model_version,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Savings vs running to failure, per model, across cutoffs (interactive line chart).
    cuts = np.unique(
        np.round(np.concatenate([np.linspace(0.002, 0.05, 25), np.linspace(0.06, 0.9, 43)]), 3)
    )
    no_model = cost_per_1000(yt, np.zeros(len(yt)), 1.0)
    savings_curve = [
        {
            "threshold": float(cut),
            **{
                name: round(1 - cost_per_1000(yt, proba(p, x_test), float(cut)) / no_model, 4)
                for name, p in fitted.items()
            },
        }
        for cut in cuts
    ]
    sweep = []
    # Include the chosen cutoff and fine low steps: with a 40x miss penalty the action is near zero.
    sweep_cuts = sorted(
        {threshold, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
    )
    for cut in sweep_cuts:
        counts = maintenance_cost(yt, test_p >= cut)
        sweep.append({"threshold": float(cut), **{k: counts[k] for k in ("tn", "fp", "fn", "tp")}})

    wear_edges = [0, 50, 100, 150, 200, 260]
    wear = pd.cut(df.tool_wear_min, bins=wear_edges, include_lowest=True)
    explore = {
        "model_comparison": [
            {"model": n, **c["test"], "threshold": c["threshold"]} for n, c in comparison.items()
        ],
        "savings_curve": savings_curve,
        "threshold_sweep": sweep,
        "feature_importance": importance_rows,
        "failure_rate_by_wear": [
            {
                "label": f"{int(iv.left) if iv.left > 0 else 0}–{int(iv.right)} min",
                "value": round(float(v), 4),
            }
            for iv, v in y.groupby(wear, observed=False).mean().items()
        ],
    }
    (REPORTS / "explore.json").write_text(json.dumps(explore, indent=2) + "\n")

    golden = [
        {
            "inputs": s["inputs"],
            "probability": round(float(proba(final, records_to_frame([s["inputs"]]))[0]), 4),
        }
        for s in samples
    ]
    (REPORTS / "golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    for name in ("metrics.json", "explore.json"):
        (artifacts / name).write_text((REPORTS / name).read_text())

    write_figures(savings_curve, list(fitted), importance_rows)
    summary = {"final": final_name, "threshold": threshold, "test": comparison[final_name]["test"]}
    print(json.dumps(summary, indent=2))


def write_figures(
    curve: list[dict[str, Any]], names: list[str], importance: list[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        ax.plot([c["threshold"] for c in curve], [c[name] * 100 for c in curve], label=name)
    ax.set_xlabel("flag a machine when failure risk is above…")
    ax.set_ylabel("cost saved vs running to failure (%)")
    ax.set_title("Choosing the cutoff by maintenance cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "savings_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = importance[::-1]
    ax.barh([r["label"] for r in top], [r["importance"] for r in top])
    ax.set_xlabel("drop in test PR-AUC when shuffled (permutation importance)")
    ax.set_title("What predicts a machine failure")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
