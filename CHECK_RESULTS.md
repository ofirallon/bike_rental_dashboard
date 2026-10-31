# Executed verification

The application was checked with Python 3.12.10 on Windows. These are executed checks, rather than an acceptance claim based only on opening the app.

## Automated checks

A clean virtual environment was installed using the submitted `requirements.txt`; `python -m pip check` reported **No broken requirements found**. The final full suite after the visual update completed with **17 passed, zero failures/errors/skips** in **33.72 seconds**. UI checks verify neutral actual-count cards against saved outcomes, outcome hiding before Reveal and resetting on model/hour changes. Temporary verification environments and intermediate outputs were subsequently removed during project cleanup.

The table below records the checks and observed outcomes. Dependencies are pinned in [requirements.txt](requirements.txt).

| Check | Observed result |
|---|---|
| Missing-hour handling | An absent t−1 stays missing despite another earlier CSV row |
| Complete 24-hour mean | Counts 1…24 give 12.5; deleting a required timestamp makes the mean missing |
| Observed zero | Zero remains zero with a false missing flag |
| Target/future boundary | Mutating target/future counts and weather leaves the current inputs and forecasts unchanged |
| Earlier observation | Mutating an eligible earlier count updates the correct input |
| Frozen preprocessing | Later mutations do not change Training medians, scales, encoding or column order |
| Entire Test changed | A separate complete training run with changed Test counts/weather produced identical fitted-model hashes, selected settings, preprocessing and default |
| Audit robustness | Invalid hours/targets and duplicate rows are excluded deterministically with reasons; observed+missing equals expected |
| Calendar convention | Saturday=6, Sunday=0 |
| MAE/RMSE example | MAE=2; RMSE=sqrt(14/3) |
| Coverage/width fixture | Inclusive endpoint gives exactly 2/3 coverage; width is checked independently |
| Poisson fixture | Mean=100 gives central 80% endpoints 87/113 and 95% endpoints 81/120; known log-probability checked |
| Bayesian independent example | Posterior mean [1.6,1.2], covariance [[.4,−.2],[−.2,.6]], future mean 4, coefficient variance 2 and total variance 3 |
| Saved-model reload | Batch validation outputs agree before/after serialization; replay agrees with saved outputs for every method at first/last and a worked Test hour |
| Worked historical inputs | 2012-08-01 08:00 uses counts 487/760/788 and 24-hour mean 300.4167 |
| Point scores and selection | Saved MAE/RMSE independently recomputed; default follows Validation selection; both Validation/Test forecasts present |
| Invalid selections | Out-of-period/unavailable targets, unknown models and mismatching saved references raise clear errors |
| Streamlit workflow | All screens load; Forecast precedes Reveal; model/hour changes clear previous results; interval controls and error inspection work |

The initial pytest attempt encountered an inaccessible system temporary directory. This was resolved by using a new project-local `--basetemp`; the workaround and copyable command are in the README. No test was skipped to obtain a passing result.

## Actual browser and exports

Chromium/Playwright verification exercised the running server, independent of Streamlit AppTest. It forecast 2012-08-01 08:00, verified the hidden outcome before Reveal and the actual **681** after Reveal, switched to Bayesian and checked result clearing, navigated all four screens, opened an error case, and downloaded both the score table and displayed forecast rows. Both downloaded CSVs matched the corresponding saved data. No JavaScript page errors were recorded.

The visual update was also exercised at 1440 × 1000 and 390 × 844: all four screens loaded, mobile navigation and stacked input tables worked, and no page-level horizontal overflow or JavaScript page errors were recorded. Checks covered interval plots, model-comparison bars, default badges and the nominal/observed/gap coverage panel.

The actual-browser [screen recording](artifacts/demo.webm) is retained and predates the latest visual update. Optional machine-readable test outputs, browser automation tools, intermediate screenshots and downloaded CSV copies were removed during project cleanup; their executed outcomes are recorded above.

The UCI download helper was also executed into a separate workspace directory. Its SHA-256 matched the original local `hour.csv`: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.

## Measured results and remaining model limitations

- Frozen default: **Ridge**, chosen by Validation MAE **50.2291**, not by Test ranking. Its Test MAE is **48.9561**, RMSE **73.4159** on 3,632 observed hours.
- Poisson no longer emits a constant prediction. Validation MAE is **83.9816**, compared with **189.2985** in the original failed fit; Test MAE is **86.6548**. Its Test mean negative log count probability is **25.7216**. It remains worse than the linear methods on point scores.
- Calendar-only Ridge Validation MAE **115.1796** versus full-input **50.2291** gives a measured difference **64.9505**. Both variants were tuned independently on the same 2,208 Validation targets.
- Training/Validation binned total variation is **0.259588**, using fixed Training-derived boundaries saved in configuration.
- Bayesian noise is estimated from Training rather than fixed at 1. Nevertheless, its Test coverage is **61.62%** for the nominal 80% interval and **74.67%** for nominal 95%. Poisson coverage is **23.43% / 34.77%**. These remain undercoverage; the app reports them rather than claiming calibration or retuning on Test.
- The source data are historical system-wide counts. This app does not claim live forecasts or station-level allocation. Gaussian negative endpoints are retained and labelled as an approximation.
- The final technical report, submitter's GitHub URL/final commit and personal review of the AI log remain submission tasks; no report, publication or Git commit was created by this repair.
