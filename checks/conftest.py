from pathlib import Path
import sys
import joblib
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


@pytest.fixture(scope='session')
def fitted(tmp_path_factory):
    from data import read_and_audit
    from train_evaluate import run_pipeline
    folder=tmp_path_factory.mktemp('reproduced_results')
    with threadpool_limits(limits=1):
        config=run_pipeline(ROOT/'hour.csv',folder)
    observed,*_=read_and_audit(ROOT/'hour.csv')
    return dict(folder=folder, config=config, bundle=joblib.load(folder/'artifacts/models.joblib'),
                observations=observed,predictions=pd.read_csv(folder/'predictions.csv'))
