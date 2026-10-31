import copy
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from forecasting import forecast_hour, reveal_outcome, verify_saved
from models import MODEL_NAMES
from evaluation import calculate_mae,calculate_rmse


def test_saved_models_replay_all_methods_and_edge_hours(fitted):
    observed,bundle,predictions=fitted['observations'],fitted['bundle'],fitted['predictions']
    test=observed.loc[observed.period.eq('Test')]
    hours=[test.timestamp.iloc[0],pd.Timestamp('2012-08-01 08:00'),test.timestamp.iloc[-1]]
    for name in MODEL_NAMES:
        for hour in hours:
            result=forecast_hour(observed,bundle,hour,name)
            verify_saved(result,predictions)
            actual=reveal_outcome(observed,result)
            saved=predictions.loc[pd.to_datetime(predictions.timestamp).eq(hour)&predictions.model.eq(name)].iloc[0]
            assert actual['actual']==saved.cnt
            assert np.isclose(actual['error'],saved.error)
    result=forecast_hour(observed,bundle,hours[1],'Ridge')
    details=result['inputs'].set_index('input')
    assert details.loc['prev_hour_cnt','raw_value']==487
    assert details.loc['prev_day_cnt','raw_value']==760
    assert details.loc['prev_week_cnt','raw_value']==788
    assert np.isclose(details.loc['prev_24h_mean','raw_value'],300.4166666667)


def test_predictor_boundary_and_frozen_saved_models(fitted):
    source,bundle=fitted['observations'],fitted['bundle']
    timestamp=pd.Timestamp('2012-08-01 08:00')
    before=joblib.hash(bundle)
    changed=source.copy()
    changed.loc[changed.timestamp.ge(timestamp),['cnt','temp','hum','windspeed']]=99999
    for name in MODEL_NAMES:
        original=forecast_hour(source,bundle,timestamp,name)
        revised=forecast_hour(changed,bundle,timestamp,name)
        assert np.isclose(original['scored_forecast'],revised['scored_forecast'])
        pd.testing.assert_frame_equal(original['inputs'],revised['inputs'])
    assert joblib.hash(bundle)==before
    changed=source.copy()
    changed.loc[changed.timestamp.eq(timestamp-pd.Timedelta(hours=1)),'cnt']=1
    modified=forecast_hour(changed,bundle,timestamp,'Ridge')
    assert modified['inputs'].set_index('input').loc['prev_hour_cnt','raw_value']==1


def test_saved_metrics_and_validation_choice(fitted):
    folder,config,predictions=fitted['folder'],fitted['config'],fitted['predictions']
    candidates=pd.read_csv(folder/'validation_results.csv')
    selected=candidates.loc[candidates.experiment.eq('principal')&candidates.selected]
    winner=min(selected.to_dict('records'),key=lambda r:(r['MAE'],MODEL_NAMES.index(r['Model'])))['Model']
    assert winner==config['default_model']
    test=predictions.loc[predictions.period.eq('Test')]
    assert set(predictions.period)=={'Validation','Test'}
    assert not predictions.duplicated(['timestamp','period','model']).any()
    metrics=pd.read_csv(folder/'metrics.csv').set_index('Model')
    for name,rows in test.groupby('model'):
        assert np.isclose(metrics.loc[name,'MAE'],calculate_mae(rows.cnt,rows.scored_forecast))
        assert np.isclose(metrics.loc[name,'RMSE'],calculate_rmse(rows.cnt,rows.scored_forecast))
    poisson=test.loc[test.model.eq('Poisson')]
    assert poisson.scored_forecast.nunique()>100
    posterior=fitted['bundle']['models']['Bayesian']
    assert np.allclose(posterior['S'],posterior['S'].T,atol=1e-8)
    assert test.loc[test.model.eq('Bayesian'),'tot_var'].ge(0).all()
    assert candidates.loc[candidates.Model.eq('Lasso')].nonzero_coefficients.notna().all()


def test_unavailable_hours_models_and_mismatching_saved_results(fitted):
    with pytest.raises(ValueError):
        forecast_hour(fitted['observations'],fitted['bundle'],'2011-01-01','Ridge')
    with pytest.raises(ValueError):
        forecast_hour(fitted['observations'],fitted['bundle'],'2012-08-01','Unknown')
    result=forecast_hour(fitted['observations'],fitted['bundle'],'2012-08-01','Ridge')
    result['scored_forecast']+=1
    with pytest.raises(ValueError,match='differs'):
        verify_saved(result,fitted['predictions'])


def test_changing_test_records_cannot_reselect_or_refit(fitted,tmp_path):
    from train_evaluate import run_pipeline
    from data import ROOT
    from threadpoolctl import threadpool_limits
    source=pd.read_csv(ROOT/'hour.csv')
    mask=pd.to_datetime(source.dteday).ge('2012-08-01')
    source.loc[mask,'cnt']=(source.loc[mask,'cnt']*1.05).round().astype(int)
    source.loc[mask,'temp']=source.loc[mask,'temp']*.95
    path=tmp_path/'changed.csv'
    source.to_csv(path,index=False)
    with threadpool_limits(limits=1):
        changed_config=run_pipeline(path,tmp_path/'changed_results')
    changed=joblib.load(tmp_path/'changed_results/artifacts/models.joblib')
    assert changed_config['preprocessing']==fitted['config']['preprocessing']
    assert changed_config['selected_settings']==fitted['config']['selected_settings']
    assert changed_config['default_model']==fitted['config']['default_model']
    for name in MODEL_NAMES:
        assert joblib.hash(changed['models'][name])==joblib.hash(fitted['bundle']['models'][name])
