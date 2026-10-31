import numpy as np
import pandas as pd
import pytest
from features import build_features, Preprocessor, HISTORY, FLAGS


def fixture():
    ts=pd.date_range('2011-01-01',periods=26,freq='h')
    return pd.DataFrame(dict(timestamp=ts,cnt=np.arange(1,27),holiday=0,workingday=0,temp=.5,hum=.6,windspeed=.1))


def test_exact_history_missing_zero_and_complete_window():
    observed=fixture()
    raw=build_features(observed)
    assert raw.iloc[24].prev_24h_mean==12.5
    removed=build_features(observed.drop(index=23))
    row=removed.loc[observed.timestamp.iloc[24]]
    assert pd.isna(row.prev_hour_cnt) and row.prev_hour_cnt_missing_flag==1
    assert pd.isna(row.prev_24h_mean)
    observed.loc[23,'cnt']=0
    raw=build_features(observed)
    assert raw.iloc[24].prev_hour_cnt==0 and raw.iloc[24].prev_hour_cnt_missing_flag==0


def test_target_and_future_outcomes_do_not_enter_predictors():
    observed=fixture()
    original=build_features(observed)
    changed=observed.copy()
    changed.loc[24:,['cnt','temp','hum','windspeed']]=99999
    result=build_features(changed)
    pd.testing.assert_series_equal(original.iloc[24],result.iloc[24])
    changed=observed.copy()
    changed.loc[23,'cnt']=999
    assert build_features(changed).iloc[24].prev_hour_cnt==999


def test_frozen_imputation_scaling_and_unknown_category():
    observed=fixture()
    raw=build_features(observed)
    prep=Preprocessor.fit(raw.iloc[:20])
    before=prep.metadata()
    mutated=observed.copy()
    mutated.loc[20:,['cnt','temp','hum','windspeed']]=99999
    trained_again=Preprocessor.fit(build_features(mutated).iloc[:20])
    assert trained_again.metadata()==before
    transformed=prep.transform(raw)
    for col in HISTORY:
        missing=raw[col].isna()
        expected=(prep.medians[col]-prep.means[col])/prep.scales[col]
        assert np.allclose(transformed.loc[missing,col],expected)
        assert transformed.loc[missing,f'{col}_missing_flag'].eq(1).all()
    unknown=raw.iloc[[0]].copy()
    unknown['month']=12
    values=prep.transform(unknown)
    assert values.filter(regex='^month=').sum(axis=1).iloc[0]==0
    assert list(values)==prep.feature_names and np.isfinite(values).all().all()
    assert prep.metadata()==before


def test_calendar_weekday_and_binary_fields():
    raw=build_features(fixture())
    assert raw.iloc[0].weekday==6  # Saturday; source convention Sunday=0.
    assert raw.iloc[24].weekday==0
    assert set(FLAGS).issubset(raw.columns)
