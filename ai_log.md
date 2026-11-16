# AI-assisted development log

These records quote actual messages from the project conversation. Prompts marked as excerpts reproduce only the relevant part. Two records follow the same visual-change request through implementation and checking; they are not presented as invented separate chats. There are two design/planning, two implementation and two checking/debugging records. Records 5 and 6 describe corrections to earlier AI proposals.

## 1. Design/planning — requirements and prediction architecture

**Task:** Identify the components required for an auditable historical replay.

**Actual prompt:**

> אתה בודק קשוח של המטלה
> תסתכל על הקריטריונים (ללא מסמך כרגע) ותגיד מה חסר ומה לא חסר
> מה לשפר ומה לא
> תעשה רשימה

**Response excerpt:**

> מצאתי בתיקייה גם את מסמך המטלה, כך שאפשר לבדוק מול הדרישות עצמן.

**Decision:** Ground the design in the assignment. Separate saved-model prediction from outcome reveal, chronological model selection from Test assessment, and software evidence from the written report.

**Code/change references:** forecasting.py: forecast_hour, reveal_outcome, verify_saved; train_evaluate.py; features.py.

**Executed verification/outcome:** The recorded 17-check suite covers prediction boundaries, frozen selection and saved-model replay. checks/test_replay.py compares all five methods against saved outputs at first/last and worked Test hours. Outcomes are recorded in CHECK_RESULTS.md.

## 2. Design/planning — interface hierarchy

**Task:** Produce a prioritized UI plan that preserves the analyst's forecast workflow.

**Actual prompt:**

> כעובד UX UI
> אני רוצה שתעשה לוג של כל הדברים בUX UI שלא טובים
> אני רוצה רשימה שאשלח למתכנת שלי

**Response excerpt:**

> אעבור על המסכים והפקדים ואכין רשימת תיקוני UX/UI שאפשר להעביר למתכנת, עם הבעיה, התיקון המוצע ורמת העדיפות.

**Decision:** Separate visual corrections from prediction behaviour. Prioritize neutral outcomes, readable input sources, explicit period scopes, measured coverage and mobile layouts. Keep Forecast-before-Reveal and the Validation-selected default.

**Code/change references:** screens/presentation.py; screens/common.py; screens/compare.py; screens/inspect.py.

**Executed verification/outcome:** Browser checks exercised all four screens at 1440 × 1000 and 390 × 844. Stacked input rows and mobile navigation worked; no page-level horizontal overflow at 390px or JavaScript page errors were recorded. The summary is retained in CHECK_RESULTS.md.

## 3. Implementation — requested visual corrections

**Task:** Implement the selected 13 visual changes and three additional charts.

**Actual prompt excerpt:**

> אתה מומחה UX UI
>
> תתקן את הדברים הוויזואלים האלה

**Response excerpt:**

> הוספתי שפה חזותית משותפת: צבעים רגועים, כותרות קומפקטיות, כרטיסי מדדים ניטרליים וטבלאות עם פירוט קריא במובייל.

**Decision:** Share presentation helpers instead of duplicating styles. Use neutral metric cards, semantic input tables and interval/model-comparison/coverage charts. Format display numbers while retaining full saved precision and unavailable interval fields.

**Code/change references:** .streamlit/config.toml; screens/presentation.py; screens/common.py: interval_chart and show_forecast; screens/compare.py; screens/inspect.py.

**Executed verification/outcome:** The full suite passed 17/17 after the visual update. The revealed actual metric matched saved outcomes. Downloaded score and forecast CSVs matched saved source rows; changing the model/hour cleared the previous forecast and outcome.

## 4. Implementation — technical report from measured evidence

**Task:** Create the report without modifying fitted models or the app.

**Actual prompt:**

> create me now the last thing that is needed for this excercise
> the PDF
> and tell me if there is something more that needs to be done
> (dont do anything else)

**Response excerpt:**

> I’ll create `report.pdf` from the current code, saved results, and assignment requirements. I’ll also list any remaining submission tasks, without changing the app or doing any other work.

**Decision:** Recompute the worked replay and Bayesian calculation from saved objects. Use three actual largest-error cases, distinguish evidence from hypotheses, and report poor interval coverage.

**Code/change references:** report.pdf; report_source.html; predictions.csv; interval_metrics.csv; artifacts/models.joblib; forecasting.py.

**Executed verification/outcome:** At 2012-08-01 08:00, Ridge predicts 664.374197 against actual 681. Bayesian mean is 664.319021 and total predictive variance is 1,187.950644. Report values were checked against saved outputs; PDF checks covered page placement, text extraction and numerical values.

## 5. Checking/debugging — reject the first chart/header layout

**Task:** Check the visual implementation from record 3 in a real browser.

**Actual prompt excerpt (the same request as record 3):**

> תתקן את הדברים הוויזואלים האלה

**Response excerpt identifying the correction:**

> בצילומים זיהיתי שני פרטים שדורשים תיקון נוסף: גרף רווחי החיזוי דחוס מדי, והכותרת העליונה נחתכת מעט.

**Rejected/corrected proposal:** The first AI implementation used a 130px interval plot and insufficient top padding. Screenshots showed overlapping interval rows and a clipped header. That proposal was rejected even though numerical/UI checks passed.

**Decision and code/change references:** Increase plot/container heights to 220/240px in screens/common.py and clear the framework header using top padding in screens/presentation.py. Do not change interval endpoints to solve a layout problem.

**Executed verification/outcome:** Repeated Chromium checks showed separated 80%/95% interval rows, a readable header and stacked mobile inputs. The final recorded full suite passed 17 tests in 33.72 seconds. Temporary screenshots were removed after visual review.

## 6. Checking/debugging — correct manual startup guidance

**Task:** Diagnose the recurring white page and provide reliable startup.

**Actual prompt:**

> fix white screen

**Response excerpt:**

> The server is stopped again; nothing is responding on port 8501. I’ll verify the app in Chrome and add a simple launcher so you can start it reliably from the project folder.

**Rejected/corrected proposal:** Earlier AI assistance supplied a valid manual Streamlit command, but instructions alone did not ensure that the server stayed running or started from the correct directory. This workflow was corrected with a directory-aware launcher. The original command itself was not syntactically or numerically wrong.

**Decision and code/change references:** start.cmd selects its own folder, uses the local environment, starts port 8501, opens the browser and keeps startup failures visible. README.md and howtorun.txt explain keeping the window open. Model code was not changed to disguise a stopped server.

**Executed verification/outcome:** Initially the health URL refused the connection and no server listened on port 8501. After startup, health returned 200; all four screens, a loaded-model forecast and page reload worked in Chrome with no JavaScript errors or failed local requests. The launcher also worked when invoked from the parent folder.
