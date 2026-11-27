"""Load the trained pipeline and explain each failure-risk prediction with what-if contributions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import skops.io as sio
from sklearn.pipeline import Pipeline

from app.features import FEATURE_ORDER, FIELDS_BY_NAME, NUMERIC_FEATURES, records_to_frame

ARTIFACT_DIR = Path(__file__).parent / "model_artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.skops"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
SAMPLES_PATH = ARTIFACT_DIR / "samples.json"

# skops refuses to load any type not explicitly trusted. These are exactly the non-default types the
# final pipeline (ColumnTransformer + OneHotEncoder + StandardScaler + the chosen model) serializes;
# anything else means the file is not the artifact we trained, so loading fails closed.
TRUSTED_TYPES = frozenset({"sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"})

# How many inputs to show in an explanation (the largest effects either way).
TOP_CONTRIBUTIONS = 6


def save(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    sio.dump(pipeline, path)


def load(path: Path = MODEL_PATH) -> Pipeline:
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - TRUSTED_TYPES
    if unexpected:
        raise RuntimeError(f"refusing to load model with unexpected types: {sorted(unexpected)}")
    return sio.load(path, trusted=sorted(untrusted))


def _value_text(name: str, value: Any) -> str:
    field = FIELDS_BY_NAME[name]
    if field.kind == "category":
        option = next(o for o in field.to_schema()["options"] if o["value"] == value)
        return f"{field.label.rstrip('?')}: {option['label']}"
    number = float(value)
    digits = 0 if float(field.step or 1).is_integer() else 1  # match the form's precision
    text = f"{number:,.{digits}f}" + (f" {field.unit}" if field.unit else "")
    return f"{field.label}: {text}"


class RepairModel:
    def __init__(self, pipeline: Pipeline, manifest: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.version: str = manifest["model_version"]
        self.threshold: float = manifest["failure_threshold"]
        self.baseline: dict[str, Any] = manifest["baseline"]

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Failure risk plus what-if contributions.

        For each reading, re-score the machine with just that reading set to a typical value
        (training median / most common variant). The change in risk is that reading's contribution;
        positive = this reading lowers the risk, negative = it raises it. One batched call.
        """
        rows = [record]
        for name in FEATURE_ORDER:
            rows.append({**record, name: self.baseline[name]})
        risks = self.pipeline.predict_proba(records_to_frame(rows))[:, 1]
        risk = float(risks[0])

        contributions: list[dict[str, Any]] = []
        for name, risk_without in zip(FEATURE_ORDER, risks[1:], strict=True):
            effect = (float(risk_without) - risk) * 100  # percentage points of failure risk
            if self._is_baseline(name, record[name]) or abs(effect) < 0.05:
                continue
            contributions.append(
                {"label": _value_text(name, record[name]), "field": name, "value": round(effect, 1)}
            )
        contributions.sort(key=lambda c: -abs(c["value"]))

        flagged = risk >= self.threshold
        return {
            "task": "classification",
            "score": round(risk, 4),
            "score_label": "Failure risk",
            "threshold": self.threshold,
            "positive": not flagged,
            "outcome": "Flag for inspection" if flagged else "Keep running",
            "explanation": {
                "kind": "contributions",
                "title": "What moved the risk",
                "value_label": (
                    "How much each reading lowers (+) or raises (−) the failure risk, in "
                    "percentage points, compared with a typical machine"
                ),
                "steps": contributions[:TOP_CONTRIBUTIONS],
            },
            "model_version": self.version,
        }

    def _is_baseline(self, name: str, value: Any) -> bool:
        base = self.baseline[name]
        if name in NUMERIC_FEATURES:
            return float(value) == float(base)
        return value == base


@lru_cache(maxsize=1)
def get_model() -> RepairModel:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return RepairModel(load(MODEL_PATH), manifest)


@lru_cache(maxsize=1)
def get_samples() -> list[dict[str, Any]]:
    return json.loads(SAMPLES_PATH.read_text())
