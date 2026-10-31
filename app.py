"""Dashboard navigation. Run: python -m streamlit run app.py"""
import hashlib
import json
import joblib
import pandas as pd
import streamlit as st
from data import ROOT, read_and_audit
from screens import overview, forecast, compare, inspect
from screens.presentation import install_styles, brand

st.set_page_config(page_title='Bike-Rental Forecast Dashboard',layout='wide')
install_styles()
TABLES=['predictions','metrics','validation_results','coefficients','history_experiment','count_distribution','interval_metrics','residual_metrics','data_audit','missing_hours','data_exclusions','baseline_groups']


@st.cache_resource
def load_bundle(stamp):
    return joblib.load(ROOT/'artifacts/models.joblib')


@st.cache_data
def load_data(stamp):
    config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    if config.get('schema_version')!=2:
        raise ValueError('Saved results use an old format. Run python train_evaluate.py.')
    digest=hashlib.sha256((ROOT/'hour.csv').read_bytes()).hexdigest()
    if digest!=config['source_sha256']:
        raise ValueError('Source data changed since training. Regenerate saved models and results.')
    observations,_,_,_=read_and_audit(ROOT/'hour.csv')
    tables={name:pd.read_csv(ROOT/f'{name}.csv') for name in TABLES}
    return observations,config,tables


brand()
try:
    files=[ROOT/'hour.csv',ROOT/'config.json',ROOT/'artifacts/models.joblib']+[ROOT/f'{name}.csv' for name in TABLES]
    stamp=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files)
    observations,config,tables=load_data(stamp)
    bundle=load_bundle(stamp)
    if bundle['run_id']!=config['run_id']:
        raise ValueError('Models and configuration belong to different runs. Regenerate results.')
except (OSError,ValueError,KeyError,EOFError) as exc:
    st.error(f'Setup required: {exc}')
    st.code('python -m pip install -r requirements.txt\npython train_evaluate.py\npython -m pytest checks\npython -m streamlit run app.py')
    st.stop()

st.sidebar.title('Navigation')
screens={'Data overview':overview.render,'Forecast one hour':forecast.render,'Compare models':compare.render,'Inspect model behaviour':inspect.render}
name=st.sidebar.radio('Go to',list(screens))
st.sidebar.markdown(f"**Default model · {config['default_model']}**")
st.sidebar.caption('Selected on Validation · fixed during Test\n\nTest: 1 Aug–31 Dec 2012')
with st.sidebar.expander('Saved model reference'):
    st.caption(config['run_id'])
screens[name](dict(observations=observations,config=config,tables=tables,bundle=bundle))
