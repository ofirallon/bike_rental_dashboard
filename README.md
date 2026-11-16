# Bike-Rental Forecast Dashboard

A historical hourly replay for the UCI Bike Sharing dataset. Models learn on 1 January 2011–30 April 2012; settings and the default are selected on 1 May–31 July 2012; frozen models are evaluated on 1 August–31 December 2012. Earlier completed test observations become valid history without retraining.

## Installation and run

Tested with **Python 3.12.10 on Windows**. Use Python 3.12 and run commands from this repository directory. The app resolves its data paths against the source directory, rather than the shell's current directory. Dependencies are pinned in [requirements.txt](requirements.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe train_evaluate.py
.\.venv\Scripts\python.exe -m pytest checks -q
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. After the initial setup, Windows users can double-click [start.cmd](start.cmd), or run `.\start.cmd` in PowerShell. The launcher selects the project directory, starts the local Python environment and opens the browser. Keep its terminal window open while using the app; **Ctrl+C** or closing that window stops the server. If startup fails, the window stays open so the error can be read. The virtual environment need not be activated. On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`; those platforms were not verified in this session.

The downloader refuses to overwrite an existing `hour.csv`; skip that command when the original file is already present. Alternatively download the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), extract `hour.csv`, and place it beside `app.py`. Attribution: UCI Machine Learning Repository, Bike Sharing dataset, original 2011–2012 Capital Bikeshare observations (17,379 hourly rows). The source file's original values and normalized weather units are retained. The source CSV is excluded from Git and is obtained using these instructions.

If the system pytest temporary directory is inaccessible, use a new project-local directory:

```powershell
$checkTemp = Join-Path (Get-Location).Path ('artifacts\check-temp-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest checks -q --basetemp $checkTemp
```

Regeneration takes approximately **3 seconds** on the development machine, excluding dependency installation. Tests also regenerate models into an isolated temporary directory. Times vary by hardware. The training command creates `artifacts/`, all result CSVs and `config.json`; it needs no preexisting generated configuration. It fits preprocessing and candidate models on Training only, chooses settings/default on Validation, saves the frozen objects, verifies reload agreement, and only then scores Test. It never refits on Training+Validation.

To start with the supplied saved outputs, install dependencies, obtain the original `hour.csv`, and run only the final Streamlit command. Source SHA-256 and model-run references are checked to prevent silently mixing results with changed source data. Regenerate if the source file differs. Fitted objects are in [artifacts/models.joblib](artifacts/models.joblib); only load the project's own model artifact.

## Workflow and code

- [app.py](app.py): navigation, cached model/result loading and source consistency checks.
- [screens/](screens/): one `render(ctx)` implementation for each screen.
- [data.py](data.py): audit, deterministic exclusion rules and chronological periods.
- [features.py](features.py): shared exact-timestamp feature construction, training-only imputation/scaling/encoding and source provenance.
- [models.py](models.py): five fitted methods and their point/distribution predictions.
- [forecasting.py](forecasting.py): single-hour model application and a separate outcome-reveal function.
- [evaluation.py](evaluation.py): outcome-based scores and diagnostics.
- [train_evaluate.py](train_evaluate.py): candidate fitting, frozen selection, saved artifacts and experiments.
- [checks/](checks/): executable numerical, time-boundary, reproducibility and Streamlit checks.

In **Forecast one hour**, choose an observed Test hour and model, press **Forecast**, inspect inputs and source times, then press **Reveal actual count**. Changing either selection clears the forecast and revealed outcome. The predictor reads target calendar fields and strictly earlier count/weather rows; it does not read the target outcome. For identical model/hour inputs, every numeric output must match the saved forecast with absolute tolerance `1e-6` and relative tolerance `0`.

**Data overview** provides source rows, missing timestamps, gap-preserving time series, hourly/workingday group sizes and a fixed Training/Validation count-distribution panel. **Compare models** separates Validation selection from full-Test scoring. **Inspect model behaviour** includes saved Lasso paths, the independently tuned Ridge input-set experiment, grouped interval diagnostics, and largest-error drilldown.

## Feature and modelling conventions

The permitted target calendar is hour, weekday (Sunday=0), month, holiday and workingday. Historical values are count at t−1/t−24/t−168, the mean of exactly 24 observed counts at t−24…t−1, and weather at t−1. Seven pre-imputation missing flags are retained. Missing hours never become artificial targets. Counts must be finite nonnegative integers; invalid targets are excluded. The first valid source row wins a duplicate timestamp, in original source order; all exclusions retain reasons and source row numbers. Invalid binary calendar fields raise a clear error.

Historical medians are learned on Training. Imputed historical values use Training means and population standard deviations; constant columns use scale 1. An entirely unavailable Training history feature uses replacement 0 and retains its missing flag. Categories are learned on Training and ordered; an unknown category gets all-zero indicators. Binary indicators remain 0/1. The four regressions share precisely this full representation; the seasonal baseline uses a frozen hour/workingday mean and an overall-mean fallback. Its group sizes are saved.

Candidate grids and selected settings are saved. Exact candidate ties keep the first listed candidate; exact method ties use `Seasonal Baseline, Ridge, Lasso, Poisson, Bayesian`. Ridge and the calendar-only diagnostic each try four positive penalties independently. Lasso tries five; its coefficient-zero tolerance is `1e-8`, excluding the intercept. Poisson tries four positive penalties and checks finite coefficients/convergence. Bayesian noise candidates derive from the Training OLS residual SSE divided by n minus design rank, with fixed multipliers 0.5/1/2 and prior precisions .001/.01/.1; Validation MAE chooses among these Training fits.

Point forecasts for Ridge/Lasso/Bayesian are clipped at zero for display and point scores, without rounding. Raw means remain saved. Poisson intervals use fixed-mean count quantiles. Bayesian intervals use the raw Gaussian mean and total variance (observation noise plus coefficient uncertainty); negative Gaussian endpoints are retained. Neither nominal interval level guarantees actual coverage. Measured undercoverage remains visible; Test results are not used to retune noise or settings.

## Saved result formats and units

| File | Contents |
|---|---|
| [config.json](config.json) | Periods, preprocessing values/column order, all selected settings, source hash, model reference, selection rules and default |
| [predictions.csv](predictions.csv) | Selected five models on every observed Validation/Test hour; raw mean, scored forecast, actual count, signed/absolute error, workingday, 80%/95% endpoints and Bayesian variances |
| [metrics.csv](metrics.csv) | Full-Test MAE/RMSE, evaluated-hour count, selected-fit time and Poisson mean negative log probability |
| [validation_results.csv](validation_results.csv) | All candidates, settings, Validation n/MAE, selected markers, Lasso nonzero counts and calendar-only experiment candidates |
| [coefficients.csv](coefficients.csv) | Named candidate coefficient paths with a separate intercept marker; Training fits only |
| [history_experiment.csv](history_experiment.csv) | Independently tuned calendar-only and full-input Ridge Validation comparison |
| [count_distribution.csv](count_distribution.csv) | Training-derived left-inclusive bins (including overflow), counts/proportions and absolute differences; TV is in config |
| [interval_metrics.csv](interval_metrics.csv) | Full-Test inclusive coverage, nominal level, mean width and n, overall/by hour/by workingday |
| [residual_metrics.csv](residual_metrics.csv) | Default-model full-Test mean signed residual and n for every hour 0–23 |
| [data_audit.csv](data_audit.csv) | Expected/observed/missing hours and duplicate/exclusion counts per period |
| [missing_hours.csv](missing_hours.csv) | Missing timestamps and their periods |
| [data_exclusions.csv](data_exclusions.csv) | Exclusion reasons and original source row numbers (header is row 1) |
| [baseline_groups.csv](baseline_groups.csv) | Frozen Training hour/workingday means and group sizes |

Counts, means, point errors and interval widths are rentals per hour; variances are squared rentals; coverage/proportions are fractions in saved CSVs (percentages when labelled in the UI); training times are seconds for the **selected candidate fit only**, excluding preprocessing, candidate search and prediction/scoring. Blank interval/variance/score cells mean unavailable, never zero. Summary rows with n=0 have unavailable statistics. Timestamps denote the dataset's local clock as supplied; no unsupported timezone conversion is introduced.

## Verification and demonstration

The interface includes neutral outcome cards, consistent number formatting, readable input-source tables, interval/model-comparison/coverage charts, explicit period labels and mobile layouts. Desktop and mobile browser verification is recorded with the checks below.

See the executed verification summary in [CHECK_RESULTS.md](CHECK_RESULTS.md) and the [AI development records](ai_log.md).

The actual-browser demo is [artifacts/demo.webm](artifacts/demo.webm), approximately three minutes, showing forecasting, input provenance, outcome reveal, model comparison, error inspection and real CSV downloads. To refresh it, run the app using the installation instructions and capture the same workflow with a screen recorder. Framework reference: [Streamlit documentation](https://docs.streamlit.io/).

Final submission still requires `report.pdf` and the repository URL/final commit or tag. These are not generated by the training command. The retained recording predates the latest visual update and should be refreshed for the final submission.
