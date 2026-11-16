"""Model fitting and outcome-free prediction distributions."""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor

MODEL_NAMES = ['Seasonal Baseline', 'Ridge', 'Lasso', 'Poisson', 'Bayesian']
INTERVAL_COLUMNS = ['lower_80', 'upper_80', 'lower_95', 'upper_95', 'obs_var', 'coef_var', 'tot_var']


def fit_seasonal_baseline(training):
    groups = training.assign(hour=training.timestamp.dt.hour).groupby(['hour', 'workingday']).cnt.agg(['mean', 'count'])
    return dict(lookup=groups['mean'].to_dict(), counts=groups['count'].to_dict(), global_mean=float(training.cnt.mean()))


def fit_ridge(x, y, alpha):
    return Ridge(alpha=alpha).fit(x, y)


def fit_lasso(x, y, alpha):
    return Lasso(alpha=alpha, max_iter=30000, tol=1e-5).fit(x, y)


def fit_poisson(x, y, alpha):
    model = PoissonRegressor(alpha=alpha, max_iter=2000, tol=1e-7).fit(x, y)
    if model.n_iter_ >= model.max_iter or not np.isfinite(model.coef_).all():
        raise RuntimeError('Poisson fit did not converge')
    if np.count_nonzero(model.coef_) == 0:
        raise RuntimeError('Poisson fit retained no predictor coefficients')
    return model


def fit_bayesian(x, y, alpha_prior=1.0, sigma2_noise=1.0):
    if alpha_prior <= 0 or sigma2_noise <= 0:
        raise ValueError('Prior precision and noise variance must be positive')
    phi = np.c_[np.ones(len(x)), np.asarray(x)]
    identity = np.eye(phi.shape[1])
    a = alpha_prior*identity + phi.T@phi/sigma2_noise
    return dict(m=np.linalg.solve(a, phi.T@np.asarray(y)/sigma2_noise), S=np.linalg.solve(a, identity), alpha=alpha_prior, sigma2=sigma2_noise)


def predict(name, model, raw, x):
    out = pd.DataFrame(index=raw.index, columns=['raw_mean', 'scored_forecast', *INTERVAL_COLUMNS], dtype=float)
    if name == 'Seasonal Baseline':
        mean = np.array([model['lookup'].get((int(h), int(w)), model['global_mean']) for h,w in zip(raw.hour, raw.workingday)])
    elif name == 'Bayesian':
        phi = np.c_[np.ones(len(x)), np.asarray(x)]
        mean = phi@model['m']
        coefficient = np.einsum('ij,jk,ik->i', phi, model['S'], phi)
        if np.min(coefficient) < -1e-8:
            raise ValueError('Negative coefficient variance')
        out['coef_var'] = np.maximum(coefficient, 0)
        out['obs_var'] = model['sigma2']
        out['tot_var'] = out.coef_var+model['sigma2']
        for level,z in [(80,1.28155),(95,1.95996)]:
            out[f'lower_{level}'] = mean-z*np.sqrt(out.tot_var)
            out[f'upper_{level}'] = mean+z*np.sqrt(out.tot_var)
    else:
        mean = model.predict(x)
    out['raw_mean'] = mean
    out['scored_forecast'] = np.maximum(mean,0) if name in ['Ridge','Lasso','Bayesian'] else mean
    if not np.isfinite(out.scored_forecast).all():
        raise ValueError('Non-finite forecast')
    if name == 'Poisson':
        for level,q in [(80,.1),(95,.025)]:
            out[f'lower_{level}'] = poisson.ppf(q,mean)
            out[f'upper_{level}'] = poisson.ppf(1-q,mean)
    return out
