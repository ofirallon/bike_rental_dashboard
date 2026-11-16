"""Train on Training, select on Validation, freeze, then evaluate Test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data import ROOT, PERIODS, read_and_audit
from features import Preprocessor, build_features
from models import MODEL_NAMES, fit_seasonal_baseline, fit_ridge, fit_lasso, fit_poisson, fit_bayesian, predict
from evaluation import calculate_mae, calculate_rmse, interval_audit, residual_summary, count_distribution, poisson_probability_score


def run_pipeline(data_path=ROOT/'hour.csv', output=ROOT):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output/'artifacts'
    artifacts.mkdir(exist_ok=True)
    observations, audit, missing, exclusions = read_and_audit(data_path)
    raw = build_features(observations)
    masks = {p: observations.period.eq(p).to_numpy() for p in PERIODS}
    prep = Preprocessor.fit(raw.loc[masks['Training']])
    x = prep.transform(raw)
    xt, xv = x.loc[masks['Training']], x.loc[masks['Validation']]
    rt, rv = raw.loc[masks['Training']], raw.loc[masks['Validation']]
    yt = observations.loc[masks['Training'], 'cnt'].to_numpy()
    yv = observations.loc[masks['Validation'], 'cnt'].to_numpy()
    candidates, coefficients, models, selected, times = [], [], {}, {}, {}
    print('Audited data; preprocessing fitted on Training only.', flush=True)

    def record(name, settings, model, fit_seconds, matrix=xt, valmatrix=xv, experiment='principal'):
        prediction = predict(name, model, rv, valmatrix).scored_forecast.to_numpy()
        score = calculate_mae(yv, prediction)
        row = dict(experiment=experiment, Model=name, Parameters=json.dumps(settings, sort_keys=True), MAE=score,
                   Evaluated_Hours=len(yv), fit_seconds=fit_seconds, selected=False, nonzero_coefficients=np.nan)
        if name in ['Ridge','Lasso','Poisson']:
            row['nonzero_coefficients'] = int(np.count_nonzero(abs(model.coef_)>1e-8))
            for feature, value in zip(matrix.columns, model.coef_):
                coefficients.append(dict(experiment=experiment, model=name, Parameters=row['Parameters'], alpha=settings['alpha'], feature=feature, coefficient=value, is_intercept=False))
            coefficients.append(dict(experiment=experiment, model=name, Parameters=row['Parameters'], alpha=settings['alpha'], feature='intercept', coefficient=float(model.intercept_), is_intercept=True))
        candidates.append(row)
        return score, row

    start = time.perf_counter()
    baseline = fit_seasonal_baseline(observations.loc[masks['Training']])
    seconds = time.perf_counter()-start
    _, baseline_row = record('Seasonal Baseline', {}, baseline, seconds)
    baseline_row['selected'] = True
    models['Seasonal Baseline'], selected['Seasonal Baseline'], times['Seasonal Baseline'] = baseline, {}, seconds
    for name, fitter, values in [('Ridge',fit_ridge,[.1,1.,10.,100.]), ('Lasso',fit_lasso,[.001,.01,.1,1.,10.]), ('Poisson',fit_poisson,[.001,.01,.1,1.])]:
        best = None
        for alpha in values:
            start = time.perf_counter()
            model = fitter(xt, yt, alpha)
            seconds = time.perf_counter()-start
            score, row = record(name, {'alpha':alpha}, model, seconds)
            if best is None or score < best[0]:
                best = (score,model,row,seconds,alpha)
        _,models[name],row,times[name],alpha = best
        selected[name] = {'alpha':alpha}
        row['selected'] = True
        print(f'{name}: Validation MAE={best[0]:.4f}, alpha={alpha}', flush=True)

    # Noise estimation uses only training OLS residuals and rank-adjusted degrees of freedom.
    phi = np.c_[np.ones(len(xt)), xt.to_numpy()]
    ols,_,rank,_ = np.linalg.lstsq(phi,yt,rcond=None)
    noise_estimate = max(float(np.sum((yt-phi@ols)**2)/max(len(yt)-rank,1)),1e-6)
    best = None
    for prior in [.001,.01,.1]:
        for multiplier in [.5,1.,2.]:
            settings = dict(alpha_prior=prior, sigma2_noise=noise_estimate*multiplier, noise_multiplier=multiplier)
            start = time.perf_counter()
            model = fit_bayesian(xt,yt,prior,settings['sigma2_noise'])
            seconds = time.perf_counter()-start
            score,row = record('Bayesian',settings,model,seconds)
            if best is None or score < best[0]:
                best = (score,model,row,seconds,settings)
    _,models['Bayesian'],row,times['Bayesian'],selected['Bayesian'] = best
    row['selected'] = True
    print(f'Bayesian: Validation MAE={best[0]:.4f}; training noise estimate={noise_estimate:.4f}',flush=True)
    principal_selected = [r for r in candidates if r['experiment']=='principal' and r['selected']]
    default = min(principal_selected,key=lambda r:(r['MAE'],MODEL_NAMES.index(r['Model'])))['Model']

    calprep = Preprocessor.fit(rt,calendar_only=True)
    calx = calprep.transform(raw)
    best = None
    for alpha in [.1,1.,10.,100.]:
        start = time.perf_counter()
        model = fit_ridge(calx.loc[masks['Training']],yt,alpha)
        seconds = time.perf_counter()-start
        score,row = record('Ridge',{'alpha':alpha},model,seconds,calx.loc[masks['Training']],calx.loc[masks['Validation']],'calendar_only')
        if best is None or score<best[0]:
            best=(score,row,alpha)
    best[1]['selected']=True
    full = next(r for r in principal_selected if r['Model']=='Ridge')
    history = pd.DataFrame([
        dict(input_set='Calendar only', selected_penalty=best[2], validation_hours=len(yv), validation_MAE=best[0], inputs='hour, weekday, month, holiday, workingday'),
        dict(input_set='Full required inputs', selected_penalty=selected['Ridge']['alpha'], validation_hours=len(yv), validation_MAE=full['MAE'], inputs='calendar + count history + previous weather + seven missing flags')])
    distribution,tv,bins = count_distribution(yt,yv)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    config = dict(schema_version=2, run_id=run_id, source_file=Path(data_path).name,
                  source_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                  periods={p.lower():dict(start=s,end=e) for p,(s,e) in PERIODS.items()},
                  preprocessing=prep.metadata(), selected_settings=selected, default_model=default,
                  model_reference='artifacts/models.joblib', prediction_tolerance=dict(atol=1e-6,rtol=0),
                  selection='lowest validation MAE; exact ties use MODEL_NAMES order; candidate ties keep first listed',
                  training_time_definition='selected candidate fit only; excludes preprocessing, search, prediction and scoring',
                  bayesian_noise_estimation=dict(method='Training OLS SSE/(n-rank); multipliers .5,1,2 and prior .001,.01,.1 selected by validation MAE',training_estimate=noise_estimate),
                  distribution=dict(bins=[float(v) if np.isfinite(v) else 'infinity' for v in bins],TV=tv),
                  model_order=MODEL_NAMES, coefficient_zero_tolerance=1e-8)
    bundle=dict(models=models,preprocessor=prep,selected=selected,run_id=run_id,default=default,source_sha256=config['source_sha256'])
    # Freeze all choices before any final-test prediction or score is computed.
    joblib.dump(bundle,artifacts/'models.joblib')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Frozen models; default={default}. Evaluating saved models.',flush=True)
    loaded=joblib.load(artifacts/'models.joblib')
    for name in MODEL_NAMES:
        before=predict(name,models[name],rv,xv).to_numpy()
        after=predict(name,loaded['models'][name],rv,xv).to_numpy()
        np.testing.assert_allclose(before,after,atol=1e-6,rtol=0,equal_nan=True)
    results=[]
    for period in ['Validation','Test']:
        mask=masks[period]
        for name in MODEL_NAMES:
            prediction=predict(name,loaded['models'][name],raw.loc[mask],x.loc[mask]).reset_index()
            prediction['period'],prediction['model']=period,name
            prediction['workingday']=observations.loc[mask,'workingday'].to_numpy()
            prediction['cnt']=observations.loc[mask,'cnt'].to_numpy()
            prediction['error']=prediction.cnt-prediction.scored_forecast
            prediction['abs_error']=abs(prediction.error)
            results.append(prediction)
    predictions=pd.concat(results,ignore_index=True)
    test=predictions.loc[predictions.period.eq('Test')]
    metrics=[]
    for name in MODEL_NAMES:
        frame=test.loc[test.model.eq(name)]
        metrics.append(dict(Model=name,MAE=calculate_mae(frame.cnt,frame.scored_forecast),RMSE=calculate_rmse(frame.cnt,frame.scored_forecast),Evaluated_Hours=len(frame),Training_Time_sec=times[name],Poisson_NLL=poisson_probability_score(frame.cnt,frame.raw_mean) if name=='Poisson' else np.nan))
    baseline_groups=pd.DataFrame([dict(hour=k[0],workingday=k[1],mean=v,n=baseline['counts'][k]) for k,v in baseline['lookup'].items()])
    outputs={'predictions.csv':predictions,'metrics.csv':pd.DataFrame(metrics),'validation_results.csv':pd.DataFrame(candidates),
             'coefficients.csv':pd.DataFrame(coefficients),'history_experiment.csv':history,'count_distribution.csv':distribution,
             'interval_metrics.csv':interval_audit(test),'residual_metrics.csv':residual_summary(test,default),
             'data_audit.csv':audit,'missing_hours.csv':missing,'data_exclusions.csv':exclusions,'baseline_groups.csv':baseline_groups}
    for filename,frame in outputs.items():
        frame.to_csv(output/filename,index=False)
    config['regeneration_seconds']=time.perf_counter()-started
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(f'Complete: {config["regeneration_seconds"]:.1f} seconds. Results in {output}',flush=True)
    return config


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'hour.csv')
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run_pipeline(args.data,args.output)
