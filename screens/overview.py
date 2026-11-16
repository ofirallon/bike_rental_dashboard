import altair as alt
import pandas as pd
import streamlit as st
from screens.common import download
from screens.presentation import TEAL, BLUE, table, scope, number, chart_style


def render(ctx):
    st.header('Data overview')
    observed, tables = ctx['observations'], ctx['tables']
    st.write('System-wide hourly rentals. A row at 08:00 describes 08:00–08:59. Missing observations are not zero-rental hours.')
    with st.expander('Field definitions and availability'):
        st.dataframe(pd.DataFrame([
            ('timestamp', 'dteday + hr; start of the recorded hour'),
            ('cnt', 'Recorded nonnegative integer rentals; revealed only after forecasting'),
            ('hour / weekday / month', 'Known target calendar; weekday Sunday=0 through Saturday=6'),
            ('holiday / workingday', 'Known target calendar binary indicators'),
            ('temp / hum / windspeed', 'Supplied normalized weather units; only completed earlier-hour values enter forecasts'),
            ('casual / registered / instant', 'Excluded from predictors: target components / row identifier')], columns=['field', 'meaning']), hide_index=True)
    scope('Audit by chronological period','Full dataset · 2011–2012','Fixed period totals; the date filter below does not change this audit.')
    audit=tables['data_audit']
    cards=st.columns(3)
    cards[0].metric('Observed hours',number(audit.observed_hours.sum(),0))
    cards[1].metric('Missing hours',number(audit.missing_hours.sum(),0))
    cards[2].metric('Duplicate / excluded rows',f'{int(audit.duplicate_rows.sum()):,} / {int(audit.excluded_rows.sum()):,}')
    with st.expander('Full audit by period'):
        table(audit)
    download(tables['data_audit'], 'Download data audit', 'data_audit.csv')
    download(tables['data_exclusions'], 'Download exclusions and reasons', 'data_exclusions.csv')
    st.divider()
    scope('Hourly observations','Selected range','The date filter applies to the time series, source rows, missing timestamps and hourly averages below.',selected=True)
    date_range = st.date_input('Overview date range', value=(observed.timestamp.min().date(), observed.timestamp.max().date()), min_value=observed.timestamp.min().date(), max_value=observed.timestamp.max().date())
    if len(date_range) == 2:
        start, end = map(pd.Timestamp, date_range)
        end += pd.Timedelta(hours=23)
        rows = observed.loc[observed.timestamp.between(start, end)].copy()
        grid = pd.date_range(start, end, freq='h')
        series = rows.set_index('timestamp').cnt.reindex(grid).rename('rentals').rename_axis('timestamp').reset_index()
        series['segment'] = series.rentals.isna().cumsum().astype(str)
        scope('Time series',f'{start:%d %b %Y} – {end:%d %b %Y}',f'Observed: {len(rows):,} · Missing: {int(series.rentals.isna().sum()):,}',selected=True)
        chart = alt.Chart(series.loc[series.rentals.notna()]).mark_line(color=TEAL).encode(x=alt.X('timestamp:T', title='Hour'), y=alt.Y('rentals:Q', title='Rentals per hour'), detail='segment:N', tooltip=['timestamp:T', alt.Tooltip('rentals:Q',format=',.0f')])
        st.altair_chart(chart_style(chart), width='stretch')
        with st.expander('Observed source rows · selected range'):
            st.dataframe(rows, hide_index=True, height=240,column_config={'cnt':st.column_config.NumberColumn('Actual rentals',format='localized'),'timestamp':st.column_config.DatetimeColumn('Hour',format='YYYY-MM-DD HH:mm')})
        missing = tables['missing_hours']
        st.write('Missing timestamps in selected range')
        st.dataframe(missing.loc[pd.to_datetime(missing.timestamp).between(start,end)], hide_index=True, height=160)
        download(missing, 'Download all missing timestamps', 'missing_hours.csv')
        if len(rows):
            groups = rows.assign(hour=rows.timestamp.dt.hour).groupby(['hour','workingday']).cnt.agg(mean_rentals='mean', n='size').reset_index()
            st.subheader('Average rentals by hour and working-day status · selected range')
            groups['Day type']=groups.workingday.map({0:'Non-working day',1:'Working day'})
            chart=alt.Chart(groups).mark_line(point=True).encode(x=alt.X('hour:O',title='Hour of day',axis=alt.Axis(labelAngle=0)),y=alt.Y('mean_rentals:Q',title='Mean rentals per hour'),color=alt.Color('Day type:N',scale=alt.Scale(range=[BLUE,TEAL])),strokeDash='Day type:N',tooltip=['hour','Day type',alt.Tooltip('mean_rentals:Q',format=',.2f'),alt.Tooltip('n:Q',title='Hours',format=',.0f')])
            st.altair_chart(chart_style(chart), width='stretch')
            with st.expander('Hourly averages · selected range'):
                table(groups.drop(columns=['workingday']))
    else:
        st.info('Select both a start date and an end date.')
    st.divider()
    st.subheader('Count distribution · fixed full Training and Validation periods')
    scope('Demand distribution','Full Training + Validation','Fixed periods · independent of the overview date filter above.')
    dist = tables['count_distribution']
    chart = dist.melt(id_vars='bin', value_vars=['training_proportion','validation_proportion'], var_name='period', value_name='proportion')
    chart['period']=chart.period.map({'training_proportion':'Training','validation_proportion':'Validation'})
    bars=alt.Chart(chart).mark_bar(cornerRadiusEnd=2).encode(x=alt.X('bin:N',sort=dist.bin.tolist(),title='Training-derived count bins',axis=alt.Axis(labelAngle=-25)), xOffset='period:N', y=alt.Y('proportion:Q',title='Share of observed hours',axis=alt.Axis(format='.0%')), color=alt.Color('period:N',scale=alt.Scale(range=[BLUE,TEAL]),title=None),tooltip=['bin:N','period:N',alt.Tooltip('proportion:Q',format='.2%')])
    st.altair_chart(chart_style(bars), width='stretch')
    st.metric('Binned total variation distance',number(ctx['config']['distribution']['TV']))
    st.caption(f"Training n={int(dist.training_count.sum())}; Validation n={int(dist.validation_count.sum())}. Bins use training quantiles, left-inclusive boundaries, and an overflow bin. They are frozen before validation.")
    table(dist.drop(columns=['lower','upper']),formats={c:lambda v:f'{100*v:.2f}%' for c in ['training_proportion','validation_proportion','absolute_difference']})
    download(dist, 'Download distribution bins and proportions', 'count_distribution.csv')
