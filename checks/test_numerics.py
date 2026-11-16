import numpy as np
import pandas as pd
from models import fit_bayesian, predict
from evaluation import calculate_mae, calculate_rmse, interval_statistics, count_distribution, poisson_probability_score


def test_point_score_example():
    y=np.array([10,20,30])
    p=np.array([12,17,31])
    assert calculate_mae(y,p)==2
    assert np.isclose(calculate_rmse(y,p),np.sqrt(14/3))


def test_inclusive_coverage_and_separate_width():
    score=interval_statistics([1,2,8],[1,1,1],[3,3,3])
    assert score['coverage']==2/3 and score['mean_width']==2 and score['n']==3
    assert np.isnan(interval_statistics([],[],[])['coverage'])


def test_bayesian_independent_small_example():
    x=pd.DataFrame({'x':[0.,1.]})
    model=fit_bayesian(x,np.array([2.,4.]))
    assert np.allclose(model['m'],[1.6,1.2])
    assert np.allclose(model['S'],[[.4,-.2],[-.2,.6]])
    assert np.allclose(model['S'],model['S'].T)
    raw=pd.DataFrame(index=pd.DatetimeIndex(['2012-08-01'],name='timestamp'))
    result=predict('Bayesian',model,raw,pd.DataFrame({'x':[2.]})).iloc[0]
    assert np.isclose(result.raw_mean,4)
    assert np.isclose(result.coef_var,2) and np.isclose(result.obs_var,1)
    assert np.isclose(result.tot_var,3)
    assert np.isclose(result.lower_80,4-1.28155*np.sqrt(3))


def test_poisson_known_quantiles_and_probability():
    class Fixed:
        def predict(self,x):
            return np.array([100.])
    raw=pd.DataFrame(index=pd.DatetimeIndex(['2012-08-01'],name='timestamp'))
    result=predict('Poisson',Fixed(),raw,pd.DataFrame({'x':[0.]})).iloc[0]
    # Independent fixed reference values, rather than calling the same ppf again.
    assert (result.lower_80,result.upper_80,result.lower_95,result.upper_95)==(87,113,81,120)
    assert np.isclose(poisson_probability_score([0],[2]),2)


def test_distribution_bins_cover_zero_maximum_and_overflow():
    train=np.array([0,0,1,2,5,10])
    validation=np.array([0,1,100])
    table,tv,bins=count_distribution(train,validation)
    assert table.training_count.sum()==len(train)
    assert table.validation_count.sum()==len(validation)
    assert np.isclose(table.training_proportion.sum(),1) and np.isclose(table.validation_proportion.sum(),1)
    assert np.isclose(tv,.5*table.absolute_difference.sum()) and 0<=tv<=1
    other,_,same=count_distribution(train,[999],bins)
    assert np.array_equal(same,bins) and other.validation_count.iloc[-1]==1
