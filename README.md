# Machine-Failure Prediction (Predictive Maintenance)

Is this machine about to fail? A **gradient-boosting model** reads six sensor values (machine
variant, air and process temperature, rotational speed, torque and tool wear) and decides whether
to inspect the machine. It's chosen by **what mistakes cost**, not by accuracy. Served as a **live
demo API** with an interactive UI.

- **Live demo:** [srikarbolleddu.github.io/ml-repair-prediction](https://srikarbolleddu.github.io/ml-repair-prediction/)
  (standalone, from `web/`), also embedded in the
  [case study on srikarbolleddu.com](https://srikarbolleddu.com/projects/repair-prediction-system)
- **Stack:** scikit-learn (logistic regression, MLP neural network, HistGradientBoosting) · FastAPI
  · AWS Lambda (container) · API Gateway · React (Vite)

> Project from the UT Austin Post-Graduate Program in AI & Machine Learning (ReneWind), rebuilt from
> scratch on an openly licensed dataset. The course's notebooks and data are not used or included.

## The cost of a mistake

Illustrative maintenance costs:

| Event | Cost |
|---|---|
| Inspection that finds nothing (false alarm) | $1,000 |
| Repair of a failure caught in time | $10,000 |
| Replacement after an unnoticed failure | $40,000 |

A missed failure costs **40× a false alarm**, so the best decision cutoff is far below 0.5. Every
model's cutoff is chosen to minimize total cost.

## Results (held-out test set, 2,000 machines, 68 failures)

| | Value |
|---|---|
| Maintenance cost saved vs running to failure | **65%** |
| Failures caught | **94%** (64 of 68) |
| False alarms | 73 per 1,000 machines |
| PR-AUC | 0.896 |
| Final model | Gradient boosting with physics features, flag when failure risk > 0.3% |

| Model (each at its cost-optimal cutoff) | Cost saved | Failures caught | PR-AUC |
|---|---|---|---|
| **Gradient boosting + physics features** | **65.2%** | 94% | **0.896** |
| Gradient boosting, raw sensors only | 63.3% | 91% | 0.855 |
| Neural network (MLP) + physics features | 58.4% | 90% | 0.668 |
| Logistic regression + physics features | 52.8% | 82% | 0.460 |

All numbers come from [`reports/metrics.json`](reports/metrics.json), which `training/train.py` writes.

### Physics features

Three features derived from the readings, with basic physics, help the model:
- **mechanical power** = torque × angular speed
- **temperature gap** = process temperature − air temperature
- **strain** = tool wear × torque

They are computed identically for training and serving in `app/features.py`. They lifted gradient
boosting's PR-AUC from 0.855 to 0.896.

### What we deliberately left out

The dataset includes five failure-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) that record
*which kind* of failure happened. They are only known after a failure, so they would leak the
answer. They are excluded, and the API rejects them.

### A note on model selection (and an honest caveat)

The first version chose the model on a single 2,000-row validation split. With only about 68
failures in it, that choice was noise: it picked the raw-sensor model. I noticed when the held-out
test set clearly favoured the physics-feature model. Selection now uses **5-fold cross-validation
on 8,000 machines** (about 270 failures), which independently picks the physics-feature model.
Because the test set prompted that change, treat the test numbers as slightly optimistic. The
cross-validated savings (65.7%) agree closely.

## The data: synthetic, and labelled as such

[AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(Matzka, 2020) is a **synthetic** dataset that its author generated to reflect real milling-machine
behaviour. It's licensed **CC BY 4.0**, and it's downloaded at training time and checksum-pinned,
not redistributed. Results show the method; a real deployment would retrain on real sensor history.

## Using the API

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/schema` | Input fields with types, ranges, units and help text (drives the demo form) |
| `GET /api/v1/samples` | Six real held-out machines (failures and healthy, including a miss and a false alarm) |
| `GET /api/v1/explore` | Chart specs: savings vs cutoff per model, threshold sweep, comparison, importance, failure rate by tool wear |
| `POST /api/v1/predict` | Failure risk, inspect/keep-running decision, and what-if contributions |
| `GET /api/v1/docs` | Interactive OpenAPI docs |

Explanations: each reading is reset to a typical value (training median or most common variant),
and the change in failure risk is that reading's contribution. It's one batched model call.

## Run locally

```bash
uv sync
uv run python training/train.py         # downloads the pinned dataset (sha256-checked) and retrains
uv run fastapi dev app/main.py          # API on http://localhost:8000/api/v1/docs
```

Demo page: `cd web && pnpm install && VITE_DEMO_API_URL=http://localhost:8000/api/v1 pnpm dev`
(without the variable it uses recorded responses in `web/src/fixture.json`; refresh them with
`uv run python scripts/export_web_fixture.py`, which a test enforces).

## Security and abuse limits (public demo)

- No secrets, and nothing visitors send is stored.
- Strict input validation (ranges, known categories, no extra fields) and a 4 KB body cap.
- Throttling: an API Gateway stage limit across all callers, plus a per-client rate limit in the app.
- Deploys use GitHub OIDC (no stored AWS keys). The deploy role trusts only this repo's
  approval-gated `production` environment.

## License

Code: see [LICENSE](LICENSE). Data: CC BY 4.0, Matzka (2020), UCI Machine Learning Repository.
