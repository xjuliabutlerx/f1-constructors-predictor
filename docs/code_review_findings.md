# Code Review Findings

**Author:** Claude Sonnet 4.6
**Branch:** `feature/model-development`
**Date:** 2026-03-14

---

## Summary

The ML architecture, data pipeline, and versioned training scripts (`v1/`, `v2/`, `v3/`) are all sound. The v3 models were trained successfully and produced accurate predictions. A few minor issues exist in the active codebase but are non-critical.

---

## Minor Issues (Non-Critical)

- **`download.py` — session loaded twice:** `download_session()` calls `session.load()` internally; the caller calls `.load()` again. Redundant but harmless.
- **`preprocess_data.py` — location parsed from filename by index:** `file.split('_')[3]` works correctly in practice because fastf1 location names contain spaces (not underscores), but is fragile if the naming convention changes.

---

## What's Working Well

- **Weighted MarginRankingLoss with midfield-focused pair sampling** (v3) is a well-reasoned improvement — midfield teams (ranks 4–7) are harder to distinguish and receive higher loss weights.
- **Learning rate decay** (`decay_rate ^ (epoch // decay_every)`) is properly implemented in v3.
- **Team name normalisation** correctly handles historical rebrands (Renault → Alpine, Force India → Racing Point → Aston Martin, Toro Rosso → AlphaTauri → RB).
- **Penalty handling** for 2018 (Force India rebrand deduction) and 2020 (Racing Point brake duct penalty) is a thoughtful domain-specific detail.
- **Data pipeline** (download → preprocess → features) is cleanly separated.
- **SHAP analysis** (`model_shap_analysis.py`) provides feature importance explainability.
- **Ensemble prediction** (`predict_ensemble.py`) averages across multiple model checkpoints for more robust final predictions.
- **Device detection** (MPS / CUDA / CPU) is correct across all versions.
