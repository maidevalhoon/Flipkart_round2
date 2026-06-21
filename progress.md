# Progress Log — Event-Driven Traffic Congestion Forecasting System

All implementation tracked here. Updated in place — append entries, do not rewrite.

---

## Session 1 — 2026-06-21

### What was done
- **Phase A & B:** Full problem deconstruction + dataset analysis (8173 rows × 46 cols confirmed).
- **Phase C:** Stakeholder decisions locked in:
  - Target = `priority` (High/Low classification)
  - Scope = all 8,173 records
  - Stack = Python + XGBoost + Streamlit, CSV-only, **Google Colab** as runtime
- **Phase D:** `implementation.md` written (living document).
- **Full codebase built:**
  - `src/config.py` — all thresholds/paths centralised
  - `src/preprocessing.py` + `tests/test_preprocessing.py`
  - `src/features.py` + `tests/test_features.py`
  - `src/split.py` (time-based, no random split)
  - `src/model.py` (Dummy baseline → LogReg → XGBoost, threshold tuned for High-recall ≥ 0.80)
  - `src/recommend.py` + `tests/test_recommend.py` (rule-based, no ML target)
  - `src/realtime_sim.py` (simulated replay, labelled clearly)
  - `app/dashboard.py` (Streamlit, 4 tabs)
  - `Traffic_Congestion_Forecasting.ipynb` (Colab main notebook, all-in-one)
  - `requirements.txt`, `.gitignore`, `README.md`
- **Git repo** linked to `https://github.com/maidevalhoon/Flipkart_round2.git`
- **First push** to GitHub: all source files committed.

### Key design decisions (from Phase B data reality)
| Decision | Reason |
|---|---|
| Classification target `priority` | Clean, 0% missing, 62/38 balance |
| Rule-based recommender (not ML) | `assigned_to_police_id` 98.4% empty; no manpower/barricade ground truth |
| Real-time = simulated replay | Static CSV; no live feed |
| Feedback loop = schema only | No predicted-vs-actual pairs exist yet |
| Time-based split | Temporal data; random split leaks future |
| CPU-only | 8k rows; XGBoost trains in <60s on Colab free tier |

### Status
- [x] Repo initialised and pushed
- [x] All source modules written
- [x] Tests written (preprocessing, features, recommend)
- [x] Colab notebook ready
- [ ] Tests run on Colab (pending: CSV upload on Colab)
- [ ] Model trained on Colab (pending)
- [ ] Dashboard launched on Colab (pending)

---

## How to continue
1. Open `Traffic_Congestion_Forecasting.ipynb` in Google Colab.
2. Run cells top-to-bottom: install deps → clone repo → upload CSV → full pipeline.
3. Record test metrics in the next entry below once Colab run completes.

---

## Session 2 — 2026-06-21 (Colab run)
- [ ] Test results (pytest pass/fail counts — still pending)
- [x] Model metrics from first run:
  - **Dummy (val)** (threshold=0.50): F1 (High): 0.7564 | PR-AUC: 0.6083
  - **LogReg (val)** (threshold=0.50): F1 (High): 1.0000 | PR-AUC: 1.0000
  - **XGBoost (val)** (threshold=0.96): F1 (High): 1.0000 | PR-AUC: 1.0000
  - **XGBoost (TEST)** (threshold=0.96): F1 (High): 0.9983 | PR-AUC: 1.0000

### ⚠️ Issue found: F1=1.0 is trivially achieved — not real learning
**Root cause:** `priority` is a policy label assigned by operators based entirely on corridor:
- `Non-corridor` → 0% High (3122/3122 events)
- Any named corridor → 99-100% High (5029/5051 events)
- A single `if corridor ≠ Non-corridor → High` achieves 99.9% accuracy.
- Corridor target-encoding in `features.py` was reconstructing the label perfectly.
- Additionally, `dup_cluster_size` was computed on full dataset before split → temporal leakage.

### Fixes applied in Session 3
See below.

---

## Session 3 — 2026-06-21 (Bug fixes)

### Changes made
1. **`src/features.py`**: Replaced target encoding of `corridor/zone/police_station` with
   **frequency encoding** (log event count per category). Removed the trivial label reconstruction.
   Also fixed `dup_cluster_size` temporal leakage — now computed on train-only, frozen for val/test.

2. **`src/model.py`**: Two-model approach:
   - **Model 1** (priority): same target, honest features → expected F1 ~0.75–0.88
   - **Model 2** (road closure): `requires_road_closure_bool` target; genuinely hard
     (8.3% TRUE vs 91.7% FALSE); more operationally direct ("does this need a barricade?")

3. **`Traffic_Congestion_Forecasting.ipynb`**: Added diagnostic cell showing the corridor
   correlation issue; updated training cells for two-model API; fixed `xgb_model` → `priority_model`
   variable name.

4. **`app/dashboard.py`**: Forecast tab now shows BOTH model outputs (severity + closure);
   Model Report tab shows metrics for both models with separate plot sections.

### Status after Session 3
- [x] Root cause of F1=1.0 identified and fixed
- [x] Two-model approach implemented
- [ ] Re-run on Colab to get honest metrics (expected: priority F1 ~0.75–0.88, closure PR-AUC ~0.50–0.70)
- [ ] Run pytest (preprocessing, features, recommend)
- [ ] Push model artifacts to GitHub
- [ ] Test dashboard

