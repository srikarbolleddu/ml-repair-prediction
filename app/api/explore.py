"""Turn the training reports into the ml-demo-kit/v1 `/explore` payload: generic chart specs that
the shared ExplorePanel renders. Captions pull their numbers from the reports, so they can never
disagree with the charts."""

from __future__ import annotations

from typing import Any


def _usd(value: float) -> str:
    return f"${value:,.0f}"


def build_explore(explore: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics["test"]
    costs = metrics["costs"]
    final = metrics["final_model"]
    models = explore["model_comparison"]
    by_name = {m["model"]: m for m in models}
    raw = by_name.get("Gradient boosting, raw sensors only")
    physics = by_name.get("Gradient boosting")
    wear = {r["label"]: r["value"] for r in explore["failure_rate_by_wear"]}
    worn = explore["failure_rate_by_wear"][-1]
    earlier = [r["value"] for r in explore["failure_rate_by_wear"][:-1]]
    ratio = costs["replacement"] // costs["false_alarm"]

    # Final model last: the slider starts where the last series peaks.
    names = sorted(by_name, key=lambda n: n == final)
    curve = [
        {"per_1000": round(point["threshold"] * 1000, 1), **{n: point[n] for n in names}}
        for point in explore["savings_curve"]
    ]
    physics_note = (
        f" Adding physics features (power, temperature gap, wear × torque) lifted gradient "
        f"boosting's PR-AUC from {raw['pr_auc']:.2f} to {physics['pr_auc']:.2f}."
        if raw and physics and physics["pr_auc"] > raw["pr_auc"]
        else ""
    )

    return {
        "contract": "ml-demo-kit/v1",
        "headline": [
            {
                "label": "Maintenance cost saved",
                "value": test["savings_vs_no_model"],
                "format": "percent",
                "hint": "vs running machines to failure",
            },
            {
                "label": "Failures caught",
                "value": test["recall"],
                "format": "percent",
                "hint": f"{test['tp']} of {test['tp'] + test['fn']} on held-out machines",
            },
            {
                "label": "False alarms",
                "value": test["false_alarms_per_1000"],
                "format": "number",
                "hint": "inspections per 1,000 machines",
            },
            {
                "label": "PR-AUC",
                "value": test["pr_auc"],
                "format": "number",
                "hint": "ranking quality for failures",
            },
        ],
        "charts": [
            {
                "id": "savings",
                "kind": "line",
                "title": "Choosing the cutoff by maintenance cost",
                "caption": (
                    f"An inspection costs {_usd(costs['false_alarm'])}, repairing a machine caught "
                    f"in time {_usd(costs['repair'])}, and replacing one that fails unnoticed "
                    f"{_usd(costs['replacement'])}: a miss costs {ratio}x a false alarm. So the "
                    "cheapest cutoff flags machines at very low risk. Drag the slider to see each "
                    "model's savings at every cutoff (held-out machines)."
                ),
                "x": {
                    "key": "per_1000",
                    "label": "Flag when failure risk exceeds (in 1,000)",
                    "scale": "log",
                },
                "y": {"label": "Cost saved vs running to failure", "format": "percent"},
                "series": [{"key": n, "label": n} for n in names],
                "points": curve,
            },
            {
                "id": "threshold",
                "kind": "threshold",
                "title": "Inspections vs missed failures",
                "caption": (
                    "Raise the cutoff and false alarms fall, but more failures slip through, and "
                    f"each miss costs {_usd(costs['replacement'])}."
                ),
                "positive_label": "Fails",
                "negative_label": "Keeps running",
                "selected": metrics["threshold"],
                "sweep": explore["threshold_sweep"],
            },
            {
                "id": "models",
                "kind": "table",
                "title": "Which model saves the most?",
                "caption": (
                    "Each model at its own cost-optimal cutoff, chosen by 5-fold cross-validation; "
                    f"held-out test results.{physics_note}"
                ),
                "columns": [
                    {"key": "model", "label": "Model"},
                    {"key": "savings_vs_no_model", "label": "Cost saved", "format": "percent"},
                    {"key": "recall", "label": "Failures caught", "format": "percent"},
                    {
                        "key": "false_alarms_per_1000",
                        "label": "False alarms / 1,000",
                        "format": "number",
                    },
                    {"key": "pr_auc", "label": "PR-AUC", "format": "number"},
                ],
                "rows": models,
                "highlight": {"key": "model", "value": final},
            },
            {
                "id": "importance",
                "kind": "bar",
                "title": "What the model relies on",
                "caption": (
                    "How much ranking quality drops when each reading is shuffled (permutation "
                    "importance, held-out machines). Derived features are computed from the "
                    "sensor readings."
                ),
                "format": "number",
                "items": [
                    {"label": f["label"], "value": f["importance"]}
                    for f in explore["feature_importance"]
                    if f["importance"] > 0
                ],
            },
            {
                "id": "rate-wear",
                "kind": "bar",
                "title": "Failure rate by tool wear",
                "caption": (
                    f"Failure rates stay between {min(earlier):.1%} and {max(earlier):.1%} until "
                    f"tools pass 200 minutes of use, then jump to {worn['value']:.1%}."
                ),
                "format": "percent",
                "reference": {"value": metrics["dataset"]["positive_rate"], "label": "Overall"},
                "items": [{"label": k, "value": v} for k, v in wear.items()],
            },
        ],
    }
