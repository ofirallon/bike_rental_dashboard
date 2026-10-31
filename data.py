"""Read and audit observations without inventing missing outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PERIODS = {'Training': ('2011-01-01', '2012-04-30'), 'Validation': ('2012-05-01', '2012-07-31'), 'Test': ('2012-08-01', '2012-12-31')}


def period_for(timestamps):
    ts = pd.Series(pd.to_datetime(timestamps), index=getattr(timestamps, 'index', None))
    result = pd.Series('Outside', index=ts.index)
    for name, (start, end) in PERIODS.items():
        result.loc[(ts >= start) & (ts < pd.Timestamp(end) + pd.Timedelta(days=1))] = name
    return result


def read_and_audit(path=ROOT / 'hour.csv'):
    df = pd.read_csv(path)
    required = {'dteday', 'hr', 'cnt', 'holiday', 'workingday', 'temp', 'hum', 'windspeed'}
    if required - set(df):
        raise ValueError(f'Missing source columns: {sorted(required - set(df))}')
    hours = pd.to_numeric(df.hr, errors='coerce')
    dates = pd.to_datetime(df.dteday, errors='coerce')
    valid_hour = hours.between(0, 23) & hours.mod(1).eq(0)
    df['timestamp'] = dates + pd.to_timedelta(hours.where(valid_hour), unit='h')
    df['period'] = period_for(df.timestamp)
    count = pd.to_numeric(df.cnt, errors='coerce')
    reason = pd.Series('', index=df.index)
    reason.loc[df.timestamp.isna()] = 'invalid timestamp/hour'
    reason.loc[reason.eq('') & (~np.isfinite(count) | count.lt(0) | count.mod(1).ne(0))] = 'invalid nonnegative integer target'
    reason.loc[reason.eq('') & df.period.eq('Outside')] = 'outside required periods'
    candidates = reason.eq('')
    duplicates = df.loc[candidates].duplicated('timestamp', keep='first')
    reason.loc[duplicates.index[duplicates]] = 'duplicate timestamp; kept first valid source row'
    exclusions = df.loc[reason.ne(''), ['timestamp', 'period']].copy()
    exclusions['source_row'] = exclusions.index + 2
    exclusions['reason'] = reason.loc[exclusions.index]
    df = df.loc[reason.eq('')].copy()
    df['cnt'] = count.loc[df.index].astype(int)
    for col in ['holiday', 'workingday']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if not df[col].isin([0, 1]).all():
            raise ValueError(f'{col} must contain known binary calendar values')
    for col in ['temp', 'hum', 'windspeed']:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    audits, missing_rows = [], []
    for period, (start, end) in PERIODS.items():
        expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq='h')
        actual = pd.DatetimeIndex(df.loc[df.period.eq(period), 'timestamp'])
        missing = expected.difference(actual)
        removed = exclusions.loc[exclusions.period.eq(period)]
        audits.append(dict(period=period, start=start, end=end, expected_hours=len(expected), observed_hours=len(actual), missing_hours=len(missing), duplicate_rows=int(removed.reason.str.startswith('duplicate').sum()), excluded_rows=len(removed)))
        missing_rows.extend({'timestamp': t, 'period': period} for t in missing)
    unknown = exclusions.loc[~exclusions.period.isin(PERIODS)]
    if len(unknown):
        audits.append(dict(period='Outside/unassigned', start='', end='', expected_hours=0, observed_hours=0, missing_hours=0, duplicate_rows=0, excluded_rows=len(unknown)))
    return df, pd.DataFrame(audits), pd.DataFrame(missing_rows, columns=['timestamp', 'period']), exclusions
