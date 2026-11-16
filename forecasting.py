"""Single-hour replay. Target outcomes are read only by reveal_outcome."""
import numpy as np
import pandas as pd
from features import BINARY, build_features, input_details
from models import predict


def forecast_hour(observations, bundle, timestamp, name):
    t = pd.Timestamp(timestamp)
    if name not in bundle['models']:
        raise ValueError('Unknown model')
    calendar = observations.loc[observations.timestamp.eq(t), ['timestamp', *BINARY]]
    if len(calendar) != 1 or not (pd.Timestamp('2012-08-01') <= t < pd.Timestamp('2013-01-01')):
        raise ValueError('Select one observed hour in the final test period')
    past = observations.loc[observations.timestamp.lt(t), ['timestamp', 'cnt', 'temp', 'hum', 'windspeed']]
    raw = build_features(past, calendar)
    prep = bundle['preprocessor']
    result = predict(name, bundle['models'][name], raw, prep.transform(raw)).iloc[0].to_dict()
    result.update(timestamp=t, model=name, model_reference=bundle['run_id'], settings=bundle['selected'][name])
    result['inputs'] = input_details(raw.iloc[0], prep, baseline=name == 'Seasonal Baseline')
    result['transformed_inputs'] = prep.transform(raw).T.rename(columns={t: 'value'}).rename_axis('feature').reset_index()
    return result


def reveal_outcome(observations, forecast):
    actual = observations.loc[observations.timestamp.eq(forecast['timestamp']), 'cnt']
    if len(actual) != 1:
        raise ValueError('Recorded outcome unavailable')
    actual = int(actual.iloc[0])
    error = actual-forecast['scored_forecast']
    return dict(actual=actual, error=error, abs_error=abs(error))


def verify_saved(forecast, predictions, atol=1e-6):
    rows = predictions.loc[pd.to_datetime(predictions.timestamp).eq(forecast['timestamp']) & predictions.model.eq(forecast['model'])]
    if len(rows) != 1:
        raise ValueError('Saved forecast reference unavailable or duplicated')
    for col in ['raw_mean', 'scored_forecast', 'lower_80', 'upper_80', 'lower_95', 'upper_95', 'obs_var', 'coef_var', 'tot_var']:
        if not np.isclose(forecast[col], rows.iloc[0][col], atol=atol, rtol=0, equal_nan=True):
            raise ValueError(f'Saved forecast differs in {col}; regenerate results')
