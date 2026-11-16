# AI development records

These six records describe assistance actually performed in this conversation. Several subtasks share the user's same implementation request; they are not presented as six separate user prompts or as a reconstruction of the student's earlier development history. Verification references are executable project checks and recorded outputs. The student should review these changes and add their own understanding/decisions before submission.

## 1. Design — assess the actual requirements

- Task/prompt: user asked in Hebrew to act as a strict assessor and identify missing criteria, excluding the technical report for now.
- Response excerpt: the assistant identified that the current interface reads predictions from CSV rather than applying a saved fitted model, and that the Lasso/history/distribution experiments were missing.
- Decision: use the local 25-page assignment as the requirements source; separate software requirements from the deferred report.
- References: `forecasting.py`, `train_evaluate.py`, `screens/`, `README.md`.
- Verification: `checks/test_app.py` exercises the forecast/reveal/reset workflow; `checks/test_replay.py` checks saved-model replay and choice preservation.

## 2. Design — split screens while sharing prediction logic

- Task/prompt: user asked whether to recommend a separate Python file for each screen, then requested step-by-step repairs.
- Response excerpt: keep `app.py` for loading/navigation and use `render()` per screen, with input construction and prediction shared between training/evaluation and the dashboard.
- Decision: four `screens/` modules use `forecasting.py`, `features.py` and `models.py`; there is no second model implementation inside a screen.
- References: `app.py`, `screens/forecast.py`, `screens/inspect.py`.
- Verification: all four screens and largest-error inspection are exercised by `checks/test_app.py`; replay values are compared with saved outputs.

## 3. Implementation — numerical preprocessing and Poisson repair

- Task/prompt: “תתקן את זה תעבור שלב שלב” applied to the supplied repair list.
- Response excerpt: add Training-only scaling and refuse a failed Poisson fit rather than hiding warnings.
- Decision: preserve raw history/source values for display; save Training medians, means, population scales, categories and column order. Use the same transformation for all four regressions.
- References: `features.Preprocessor`, `models.fit_poisson`, `config.json`.
- Verification: the original diagnostic found one Poisson iteration, zero nonzero coefficients and Validation MAE 189.2985; a scaled diagnostic achieved 83.9642. The repaired saved model has Validation MAE about 83.9816 and varying predictions. `test_frozen_imputation_scaling_and_unknown_category` and `test_saved_metrics_and_validation_choice` check the final behavior.

## 4. Implementation — experiments and uncertainty outputs

- Task/prompt: same repair request, including missing Lasso/Ridge/distribution experiments and Bayesian variance display.
- Response excerpt: training freezes settings before Test, saves fitted objects, and only then evaluates Test.
- Decision: independently tune calendar-only Ridge; retain Lasso paths/counts; freeze Training-derived count bins. Estimate the Bayesian noise scale using Training residuals and select candidate configurations using Validation MAE, without Test calibration.
- References: `train_evaluate.run_pipeline`, `evaluation.py`, `screens/inspect.py`, `screens/overview.py`.
- Verification: `test_distribution_bins_cover_zero_maximum_and_overflow`, the independent Bayesian small example, and `test_changing_test_records_cannot_reselect_or_refit` verify these boundaries. `interval_metrics.csv` records measured coverage rather than claiming nominal coverage was achieved.

## 5. Checking/debugging — correct the narrow initial verification

- Task/prompt: the user's strict assessment request followed the earlier server setup and screen-loading checks.
- Response excerpt: a successful application launch alone does not establish compliance or correct uncertainty.
- Corrected proposal/decision: the earlier successful screen-loading check was insufficient as acceptance evidence. Replace that narrow acceptance standard with independent arithmetic fixtures, historical-availability mutation checks, save/reload checks, actual model application, and browser/export checks.
- References: `checks/test_features.py`, `checks/test_numerics.py`, `checks/test_replay.py`, `CHECK_RESULTS.md`. The optional browser automation helper used during development was removed during submission cleanup.
- Verification: the exact 24-hour mean is 12.5; missing timestamps stay missing; observed zero stays zero; changing target/future count and weather does not alter the current forecast; Bayesian example gives mean 4 and total variance 3. The test suite records these checks explicitly.

## 6. Checking/debugging — correct the test temporary-directory choice

- Task/prompt: same step-by-step repair request, specifically executable checks and reproducibility.
- Response excerpt: run pytest with a fresh temporary directory inside the project after the system pytest directory proved inaccessible.
- Corrected proposal/decision: the first test command assumed the default Windows pytest temp location was usable. It produced 11 passes and 5 setup errors caused by `PermissionError`, not model failures. Correct the invocation to use a unique project-local `--basetemp`; retain the normal portable invocation in the README and document the fallback.
- References: `README.md`, `checks/conftest.py`, `CHECK_RESULTS.md`.
- Verification: the corrected invocation completed the initial 16 checks with no failures or warnings; the final expanded run is recorded in `CHECK_RESULTS.md`. Optional machine-readable test outputs were removed during submission cleanup.
