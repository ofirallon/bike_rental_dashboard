import altair as alt
import pandas as pd
import streamlit as st
from forecasting import reveal_outcome
from screens.presentation import NAVY, number, table, scope, chart_style, model_details


def download(frame, label, filename):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8'), filename, 'text/csv', key=filename)


def interval_chart(result, actual=None):
    intervals = pd.DataFrame([
        {'level':f'{level}% interval','lower':result[f'lower_{level}'],'upper':result[f'upper_{level}']}
        for level in [95,80]
    ])
    bands = alt.Chart(intervals).mark_rule(strokeWidth=9,color='#9bb8cd').encode(
        x=alt.X('lower:Q',title='Rentals per hour',scale=alt.Scale(zero=False)), x2='upper:Q',
        y=alt.Y('level:N',sort=['95% interval','80% interval'],title=None),
        tooltip=['level:N',alt.Tooltip('lower:Q',format=',.2f'),alt.Tooltip('upper:Q',format=',.2f')])
    points = [{'level':level,'series':'Point forecast','rentals':result['scored_forecast']} for level in intervals.level]
    if actual is not None:
        points += [{'level':level,'series':'Recorded actual','rentals':actual['actual']} for level in intervals.level]
    series=['Point forecast','Recorded actual'] if actual is not None else ['Point forecast']
    markers = alt.Chart(pd.DataFrame(points)).mark_point(size=110,filled=True,color=NAVY).encode(
        x='rentals:Q', y=alt.Y('level:N',sort=['95% interval','80% interval']),
        shape=alt.Shape('series:N',title=None,scale=alt.Scale(domain=series,range=['diamond','circle'][:len(series)])),
        tooltip=['series:N',alt.Tooltip('rentals:Q',format=',.2f')])
    st.altair_chart(chart_style((bands+markers).properties(height=220)),width='stretch',height=240)


def show_forecast(result, observations, reveal=False, interval_metrics=None):
    st.subheader(f"{result['model']} · {result['timestamp']:%Y-%m-%d %H:%M} · Test")
    actual = reveal_outcome(observations,result) if reveal else None
    if reveal:
        columns = st.columns(4)
        for col,label,value in zip(columns,
            ['Predicted rentals during this hour','Recorded actual count','Signed error','Absolute error'],
            [number(result['scored_forecast']),number(actual['actual'],0),f"{actual['error']:+,.2f}",number(actual['abs_error'])]):
            col.metric(label,value)
        st.caption('Signed error = actual − forecast. Positive means underprediction; negative means overprediction.')
    else:
        st.metric('Predicted rentals during this hour',number(result['scored_forecast']))
    if pd.notna(result['lower_80']):
        interval_chart(result,actual)
        bounds = st.columns(2)
        for col,level in zip(bounds,[80,95]):
            col.markdown(f"**{level}% prediction interval**  \n{number(result[f'lower_{level}'])} to {number(result[f'upper_{level}'])} rentals")
        if interval_metrics is not None:
            measured=interval_metrics.loc[interval_metrics.model.eq(result['model']) & interval_metrics.grouping.eq('overall')]
            scope('Measured interval coverage','Full Test · 1 Aug–31 Dec 2012','Nominal probability does not guarantee observed coverage.')
            comparison=pd.DataFrame([{'Nominal target':f'{int(r.level)}%','Observed coverage':f'{100*r.coverage:.2f}%',
                'Gap to target':f'{100*r.coverage-r.level:+.2f} pp'} for r in measured.itertuples()])
            table(comparison)
            if any(100*r.coverage<r.level for r in measured.itertuples()):
                st.html('<p class="coverage-note">Below nominal target on the full Test period. Interpret these ranges alongside the measured coverage, not as guaranteed reliability.</p>')
    else:
        st.caption('Prediction intervals unavailable for this method.')
    if result['model'] == 'Bayesian':
        table(pd.DataFrame({'Variance component': ['Observation noise', 'Coefficient uncertainty', 'Total predictive'],
                           'Variance (rentals²)': [result['obs_var'], result['coef_var'], result['tot_var']]}))
        st.caption('Gaussian approximation: intervals retain their raw mean and any negative endpoints; only the displayed point forecast is clipped at zero.')
    model_details(result)
    st.write('Inputs used')
    inputs=result['inputs'].copy()
    names={'hour':'Hour','weekday':'Weekday (Sunday = 0)','month':'Month','holiday':'Holiday','workingday':'Working day',
           'prev_hour_cnt':'Previous-hour count','prev_day_cnt':'Previous-day count','prev_week_cnt':'Previous-week count',
           'prev_24h_mean':'Previous-24-hour mean','prev_hour_temp':'Previous-hour temperature',
           'prev_hour_hum':'Previous-hour humidity','prev_hour_wind':'Previous-hour wind speed'}
    inputs['input']=inputs.input.map(lambda v:names.get(v,v))
    inputs['source']=inputs.source.str.replace(' to ','\nto\n',regex=False).str.replace(' (24 complete hours)','\n24 complete hours',regex=False)
    table(inputs,labels={'input':'Input','raw_value':'Observed value','source':'Source hour / range','missing':'Missing?',
                        'replacement':'Training replacement','model_value':'Value after imputation','used':'Used?'},
          formats={c:lambda v:number(v,4 if 0<abs(v)<1 else 2) for c in ['raw_value','replacement','model_value']})
    st.caption('Weather remains in the supplied normalized units. Missing flags refer to raw availability. model_value is the value after imputation and before scaling. The baseline uses only hour and workingday; other context is marked unused.')
    if result['model'] != 'Seasonal Baseline':
        with st.expander('Encoded and scaled model inputs (including missing flags)'):
            table(result['transformed_inputs'],labels={'value':'Scaled value'},formats={'value':4})
