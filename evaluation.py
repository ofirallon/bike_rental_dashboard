"""Scores and diagnostics, separate from outcome-free prediction."""
import numpy as np
import pandas as pd
from scipy.stats import poisson


def calculate_mae(y, forecast):
    return float(np.mean(np.abs(np.asarray(y)-np.asarray(forecast))))


def calculate_rmse(y, forecast):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(forecast))**2)))


def interval_statistics(actual, lower, upper):
    actual, lower, upper = map(np.asarray, (actual, lower, upper))
    if not len(actual):
        return dict(n=0, coverage=np.nan, mean_width=np.nan)
    return dict(n=len(actual), coverage=float(np.mean((lower <= actual) & (actual <= upper))),
                mean_width=float(np.mean(upper-lower)))


def interval_audit(predictions):
    rows = []
    for name in ['Poisson', 'Bayesian']:
        frame = predictions.loc[predictions.model.eq(name)].copy()
        frame['hour'] = pd.to_datetime(frame.timestamp).dt.hour
        for level in [80, 95]:
            groups = [('overall', 'all', frame)]
            groups += [('hour', str(hour), frame.loc[frame.hour.eq(hour)]) for hour in range(24)]
            groups += [('workingday', str(w), frame.loc[frame.workingday.eq(w)]) for w in [0, 1]]
            for kind, group, values in groups:
                rows.append(dict(model=name, level=level, grouping=kind, group=group, nominal=level/100,
                                 **interval_statistics(values.cnt, values[f'lower_{level}'], values[f'upper_{level}'])))
    return pd.DataFrame(rows)


def residual_summary(predictions, default):
    rows = predictions.loc[predictions.model.eq(default)].copy()
    rows['hour'] = pd.to_datetime(rows.timestamp).dt.hour
    summary = rows.groupby('hour').error.agg(mean_residual='mean', n='size').reindex(range(24))
    summary['n'] = summary.n.fillna(0).astype(int)
    return summary.reset_index().assign(model=default, period='Test')


def count_distribution(training, validation, bins=None):
    # Bins are chosen from training alone, then frozen before inspecting validation.
    if bins is None:
        finite = np.unique(np.r_[0., np.quantile(training, [.2, .4, .6, .8, 1.])])
        bins = np.r_[finite, np.inf]
    bins = np.asarray(bins, dtype=float)
    p, _ = np.histogram(training, bins=bins)
    q, _ = np.histogram(validation, bins=bins)
    table = pd.DataFrame(dict(lower=bins[:-1], upper=bins[1:], training_count=p, validation_count=q,
                              training_proportion=p/len(training), validation_proportion=q/len(validation)))
    table['absolute_difference'] = abs(table.training_proportion-table.validation_proportion)
    table['bin'] = [f'[{lo:g}, {hi:g})' for lo,hi in zip(bins[:-1], bins[1:])]
    return table, float(table.absolute_difference.sum()/2), bins


def poisson_probability_score(actual, mean):
    return float(-poisson.logpmf(np.asarray(actual), np.asarray(mean)).mean())
