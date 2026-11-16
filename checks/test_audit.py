import pandas as pd
from data import read_and_audit


def test_invalid_targets_duplicates_and_missing_hours(tmp_path):
    frame=pd.DataFrame(dict(dteday=['2011-01-01']*5,hr=[0,0,1,2,25],cnt=[0,10,-1,1.5,8],holiday=0,workingday=0,temp=.5,hum=.6,windspeed=.1))
    path=tmp_path/'source.csv'
    frame.to_csv(path,index=False)
    observed,audit,missing,excluded=read_and_audit(path)
    assert len(observed)==1 and observed.cnt.iloc[0]==0
    assert len(excluded)==4
    assert audit.duplicate_rows.sum()==1
    assert pd.Timestamp('2011-01-01 01:00') in missing.timestamp.tolist()
    assert (audit.expected_hours==audit.observed_hours+audit.missing_hours).all()
