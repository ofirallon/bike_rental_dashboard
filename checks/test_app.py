from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'


def actual_metrics(app):
    return [metric for metric in app.metric if metric.label=='Recorded actual count']


def test_screens_and_forecast_reveal_reset():
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    assert not app.exception
    for screen in ['Forecast one hour','Compare models','Inspect model behaviour','Data overview']:
        app.sidebar.radio[0].set_value(screen).run()
        assert not app.exception,[e.message for e in app.exception]
    app.sidebar.radio[0].set_value('Forecast one hour').run()
    assert not app.metric
    app.button[0].click().run()
    assert not app.exception and app.metric
    assert not actual_metrics(app)
    next(b for b in app.button if b.label=='Reveal actual count').click().run()
    assert actual_metrics(app) and not app.success
    saved=pd.read_csv(APP.parent/'predictions.csv',parse_dates=['timestamp'])
    reference=saved.loc[saved.model.eq(app.selectbox[1].value) & saved.timestamp.eq(app.selectbox[0].value)].iloc[0]
    assert actual_metrics(app)[0].value==f'{int(reference.cnt):,}'
    app.selectbox[1].set_value('Bayesian').run()
    assert not actual_metrics(app) and not app.metric
    next(b for b in app.button if b.label=='Forecast').click().run()
    assert not app.exception
    app.selectbox[0].set_value(pd.Timestamp(app.selectbox[0].options[1])).run()
    assert not actual_metrics(app) and not app.metric


def test_error_inspection_and_interval_selectors():
    app=AppTest.from_file(str(APP),default_timeout=60).run()
    app.sidebar.radio[0].set_value('Inspect model behaviour').run()
    by_label={s.label:s for s in app.selectbox}
    by_label['Interval model'].set_value('Bayesian').run()
    by_label={s.label:s for s in app.selectbox}
    by_label['Nominal interval level'].set_value(95).run()
    by_label={s.label:s for s in app.selectbox}
    by_label['Inspect error hour'].set_value(by_label['Inspect error hour'].options[0]).run()
    assert not app.exception and actual_metrics(app) and not app.success
