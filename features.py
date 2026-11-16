"""Shared, timestamp-based inputs and frozen training-only preprocessing."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

HISTORY = ['prev_hour_cnt', 'prev_day_cnt', 'prev_week_cnt', 'prev_24h_mean', 'prev_hour_temp', 'prev_hour_hum', 'prev_hour_wind']
CATEGORIES = ['hour', 'weekday', 'month']
BINARY = ['holiday', 'workingday']
FLAGS = [f'{col}_missing_flag' for col in HISTORY]
SOURCES = {'prev_hour_cnt': ('cnt', 1), 'prev_day_cnt': ('cnt', 24), 'prev_week_cnt': ('cnt', 168), 'prev_hour_temp': ('temp', 1), 'prev_hour_hum': ('hum', 1), 'prev_hour_wind': ('windspeed', 1)}


def build_features(observations, calendar=None):
    calendar = observations[['timestamp', *BINARY]] if calendar is None else calendar
    ts = pd.DatetimeIndex(calendar.timestamp)
    if not len(ts):
        raise ValueError('No target hours selected')
    raw = pd.DataFrame(index=ts)
    raw.index.name = 'timestamp'
    raw['hour'], raw['weekday'], raw['month'] = ts.hour, (ts.day_of_week + 1) % 7, ts.month
    for col in BINARY:
        raw[col] = calendar[col].to_numpy()
    observed = observations.set_index('timestamp').sort_index()
    for name, (field, lag) in SOURCES.items():
        raw[name] = observed[field].reindex(ts - pd.Timedelta(hours=lag)).to_numpy()
    if len(observed):
        grid = pd.date_range(min(observed.index.min(), ts.min()-pd.Timedelta(hours=24)), max(observed.index.max(), ts.max()), freq='h')
        hourly = observed.cnt.reindex(grid)
        raw['prev_24h_mean'] = hourly.rolling(24, min_periods=24).mean().shift(1).reindex(ts).to_numpy()
    else:
        raw['prev_24h_mean'] = np.nan
    raw[FLAGS] = raw[HISTORY].isna().astype(int).to_numpy()
    return raw


@dataclass
class Preprocessor:
    medians: dict
    means: dict
    scales: dict
    categories: dict
    feature_names: list
    calendar_only: bool = False

    @classmethod
    def fit(cls, training, calendar_only=False):
        medians = training[HISTORY].median().fillna(0).to_dict()
        filled = training[HISTORY].fillna(medians)
        means = filled.mean().to_dict()
        scales = filled.std(ddof=0).replace(0, 1).fillna(1).to_dict()
        categories = {col: sorted(int(v) for v in training[col].unique()) for col in CATEGORIES}
        names = BINARY + [f'{col}={v:02d}' for col in CATEGORIES for v in categories[col]]
        if not calendar_only:
            names += HISTORY + FLAGS
        return cls(medians, means, scales, categories, names, calendar_only)

    def transform(self, raw):
        x = raw[BINARY].astype(float).copy()
        for col, values in self.categories.items():
            for value in values:
                x[f'{col}={value:02d}'] = raw[col].eq(value).astype(float)
        if not self.calendar_only:
            for col in HISTORY:
                x[col] = (raw[col].fillna(self.medians[col])-self.means[col])/self.scales[col]
            x[FLAGS] = raw[FLAGS].astype(float)
        return x[self.feature_names]

    def metadata(self):
        return dict(medians=self.medians, means=self.means, scales=self.scales, categories=self.categories, feature_names=self.feature_names, calendar_only=self.calendar_only, constant_column_rule='scale=1', unknown_category_rule='all saved indicators=0', all_missing_training_rule='median=0; missing flag retained')


def input_details(raw_row, preprocessor, baseline=False):
    t = pd.Timestamp(raw_row.name)
    rows = []
    for col in CATEGORIES+BINARY:
        rows.append(dict(input=col, raw_value=raw_row[col], source='known target calendar', missing=False, replacement=np.nan, model_value=raw_row[col], used=(col in ['hour', 'workingday'] if baseline else True)))
    for col in HISTORY:
        source = f'{t-pd.Timedelta(hours=24)} to {t-pd.Timedelta(hours=1)} (24 complete hours)' if col == 'prev_24h_mean' else str(t-pd.Timedelta(hours=SOURCES[col][1]))
        missing = bool(raw_row[f'{col}_missing_flag'])
        replacement = preprocessor.medians[col] if missing else np.nan
        rows.append(dict(input=col, raw_value=raw_row[col], source=source, missing=missing, replacement=replacement, model_value=replacement if missing else raw_row[col], used=not baseline))
    return pd.DataFrame(rows)
