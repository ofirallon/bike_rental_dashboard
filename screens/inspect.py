import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from forecasting import forecast_hour, verify_saved
from screens.common import download, show_forecast
from screens.presentation import TEAL, NAVY, BLUE, AMBER, table, scope, number, chart_style, coverage_summary


def render(ctx):
    st.header('Inspect model behaviour')
    st.caption('Explore saved experiments and diagnostics. Each panel labels its own evaluation period.')
    tables=ctx['tables']
    regularisation,history,coverage,errors=st.tabs(['Regularisation','History experiment','Interval coverage','Largest errors and residuals'])
    with regularisation:
        scope('Lasso regularisation','Full Validation · 1 May–31 July 2012','All candidates fitted on Training. The marked penalty was selected using Validation MAE.')
        rows=tables['validation_results']
        rows=rows.loc[rows.experiment.eq('principal') & rows.Model.eq('Lasso')].copy()
        rows['alpha']=rows.Parameters.map(lambda s:json.loads(s)['alpha'])
        chosen=ctx['config']['selected_settings']['Lasso']['alpha']
        plot_columns=st.columns(2)
        for container,(col,label) in zip(plot_columns,[('MAE','Validation MAE (rentals/hour)'),('nonzero_coefficients','Nonzero predictor coefficients')]):
            low,high=float(rows[col].min()),float(rows[col].max())
            padding=max((high-low)*.2,.5)
            scale=alt.Scale(domain=[low-padding,high+2*padding],zero=False) if col=='MAE' else alt.Scale(domain=[0,high*1.2])
            base=alt.Chart(rows).encode(x=alt.X('alpha:Q',scale=alt.Scale(type='log'),axis=alt.Axis(values=rows.alpha.tolist()),title='Lasso penalty α'),y=alt.Y(f'{col}:Q',title=label,scale=scale),tooltip=[alt.Tooltip('alpha:Q',title='Penalty'),alt.Tooltip(f'{col}:Q',format=',.2f' if col=='MAE' else ',.0f'),alt.Tooltip('Evaluated_Hours:Q',title='Validation hours',format=',.0f')])
            selected_rows=rows.loc[rows.alpha.eq(chosen)].copy()
            selected_rows['annotation']=f'Selected α = {chosen:g}'
            selected_base=alt.Chart(selected_rows).encode(x='alpha:Q',y=f'{col}:Q')
            rule=alt.Chart(pd.DataFrame({'alpha':[chosen]})).mark_rule(color=TEAL,strokeDash=[4,4]).encode(x='alpha:Q')
            dot=selected_base.mark_point(size=125,filled=True,color=TEAL,stroke=NAVY,strokeWidth=1)
            text=selected_base.mark_text(align='left',dx=9,dy=-15,fontWeight='bold',color=TEAL).encode(text='annotation:N')
            with container:
                st.altair_chart(chart_style((base.mark_line(point=True,color=BLUE)+rule+dot+text).properties(height=240)),width='stretch')
        st.caption(f'Selected penalty: {chosen:g} · labelled dot and dashed line. MAE uses a focused, non-zero axis to show differences. Nonzero means abs(coefficient)>1e-8; intercept excluded.')
        penalty=st.selectbox('Lasso penalty coefficients',rows.alpha.tolist(),index=rows.alpha.tolist().index(chosen))
        coef=tables['coefficients']
        coef=coef.loc[coef.model.eq('Lasso') & coef.experiment.eq('principal')]
        chosen_coef=coef.loc[coef.alpha.eq(penalty)]
        st.write('Intercept: '+number(float(chosen_coef.loc[chosen_coef.is_intercept,'coefficient'].iloc[0])))
        table(chosen_coef.loc[~chosen_coef.is_intercept,['feature','coefficient']])
        st.write('Named predictor coefficients across tested penalties')
        st.dataframe(coef.loc[~coef.is_intercept].pivot(index='feature',columns='alpha',values='coefficient').style.format('{:,.2f}'))
        download(coef,'Download Lasso coefficient paths','lasso_coefficients.csv')
    with history:
        scope('Value of earlier information','Full Validation · 1 May–31 July 2012','Both Ridge input sets tuned independently on the same observed Validation hours.')
        comparison=tables['history_experiment']
        table(comparison)
        gain=float(comparison.iloc[0].validation_MAE-comparison.iloc[1].validation_MAE)
        st.metric('Calendar-only MAE minus full-input MAE',number(gain))
        st.caption('Each input set is tuned independently on the same Validation hours. A positive difference favors the combined history/weather inputs; it does not isolate a causal contribution.')
        download(comparison,'Download history experiment','history_experiment.csv')
    with coverage:
        scope('Interval coverage','Full Test · 1 Aug–31 Dec 2012','Fixed full-Test audit · no date filter is applied.')
        model_column,level_column=st.columns(2)
        name=model_column.selectbox('Interval model',['Poisson','Bayesian'])
        level=level_column.selectbox('Nominal interval level',[80,95],format_func=lambda x:f'{x}%')
        audit=tables['interval_metrics']
        audit=audit.loc[audit.model.eq(name) & audit.level.eq(level)].copy()
        overall=audit.loc[audit.grouping.eq('overall')].iloc[0]
        coverage_summary(overall,level)
        st.caption(f'{name} · full Test · n={int(overall.n):,}. Inclusive endpoints; saved distribution endpoints retained, including negative Bayesian bounds.')
        hourly=audit.loc[audit.grouping.eq('hour')].copy()
        hourly['hour']=pd.to_numeric(hourly['group'])
        hourly['Observed coverage']=100*hourly.coverage
        hourly['Position']=np.where(hourly['Observed coverage']<level,'Below nominal target','At or above nominal target')
        bars=alt.Chart(hourly).mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X('hour:O',title='Hour of day',sort=list(range(24)),axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Observed coverage:Q',title='Observed coverage (%)',scale=alt.Scale(domain=[0,100])),
            color=alt.Color('Position:N',scale=alt.Scale(domain=['Below nominal target','At or above nominal target'],range=[AMBER,BLUE]),title=None),
            tooltip=[alt.Tooltip('hour:O',title='Hour'),alt.Tooltip('Observed coverage:Q',format='.2f'),alt.Tooltip('nominal:Q',title='Nominal target',format='.0%'),alt.Tooltip('mean_width:Q',title='Mean width (rentals)',format=',.2f'),alt.Tooltip('n:Q',title='Observed hours',format=',.0f')])
        target=alt.Chart(pd.DataFrame({'target':[level]})).mark_rule(color=NAVY,strokeDash=[6,4],strokeWidth=2).encode(y='target:Q')
        st.altair_chart(chart_style((bars+target).properties(height=245)),width='stretch')
        st.caption(f'Dashed reference: {level}% nominal target. Bar position shows coverage; group sizes and interval widths are available in the tooltip and tables.')
        for grouping in ['hour','workingday']:
            st.write(f'By {grouping}')
            display=audit.loc[audit.grouping.eq(grouping)].copy()
            display['coverage_percent']=100*display.coverage
            display['nominal_percent']=100*display.nominal
            if grouping=='workingday':
                display['group']=display['group'].map({'0':'Non-working day','1':'Working day'})
            table(display[['group','n','coverage_percent','nominal_percent','mean_width']],labels={'group':'Hour of day' if grouping=='hour' else 'Day type'})
        download(audit,'Download selected interval audit','selected_interval_audit.csv')
    with errors:
        scope('Largest errors and hourly residuals','Full Test · 1 Aug–31 Dec 2012','Uses the frozen default selected on Validation, not the best model chosen after inspecting Test.')
        name=ctx['config']['default_model']
        predictions=tables['predictions']
        test=predictions.loc[predictions.period.eq('Test') & predictions.model.eq(name)]
        largest=test.sort_values(['abs_error','timestamp'],ascending=[False,True],kind='stable').head(10).reset_index(drop=True)
        st.subheader(f'Ten largest absolute errors · {name} · Test')
        error_columns=['timestamp','model','cnt','scored_forecast','error','abs_error']
        event=st.dataframe(largest[error_columns],hide_index=True,on_select='rerun',selection_mode='single-row',key='error_table',
            column_config={'timestamp':st.column_config.TextColumn('Hour',width='medium'),'model':'Model','cnt':st.column_config.NumberColumn('Actual rentals',format='localized'),
                'scored_forecast':st.column_config.NumberColumn('Forecast',format='%.2f'),'error':st.column_config.NumberColumn('Signed error',format='%.2f'),
                'abs_error':st.column_config.NumberColumn('Absolute error',format='%.2f')})
        with st.expander('Read full error values · mobile-friendly cards'):
            table(largest[error_columns])
        selected=st.selectbox('Inspect error hour',largest.timestamp.tolist(),index=None,placeholder='Select a case or click a table row')
        if event.selection.rows:
            selected=largest.iloc[event.selection.rows[0]].timestamp
        if selected is not None:
            result=forecast_hour(ctx['observations'],ctx['bundle'],selected,name)
            verify_saved(result,predictions)
            show_forecast(result,ctx['observations'],reveal=True,interval_metrics=tables['interval_metrics'])
        residuals=tables['residual_metrics']
        st.subheader(f'Mean signed residual by hour · {name} · Test')
        residuals=residuals.copy()
        residuals['Direction']=np.where(residuals.mean_residual>=0,'Underprediction (+)','Overprediction (−)')
        st.caption('Above zero (+): underprediction · Below zero (−): overprediction. Signed errors can cancel; these are not absolute errors.')
        bars=alt.Chart(residuals).mark_bar(cornerRadiusEnd=2).encode(x=alt.X('hour:O',title='Target hour',axis=alt.Axis(labelAngle=0)),y=alt.Y('mean_residual:Q',title='Actual minus point forecast'),
            color=alt.Color('Direction:N',scale=alt.Scale(domain=['Underprediction (+)','Overprediction (−)'],range=[TEAL,AMBER]),title=None),
            tooltip=[alt.Tooltip('hour:O',title='Hour'),'Direction:N',alt.Tooltip('mean_residual:Q',title='Mean signed error',format=',.2f'),alt.Tooltip('n:Q',title='Observed hours',format=',.0f')])
        zero=alt.Chart(pd.DataFrame({'zero':[0]})).mark_rule(color=NAVY,strokeWidth=2).encode(y='zero:Q')
        st.altair_chart(chart_style(bars+zero),width='stretch')
        st.caption('Positive means underprediction; negative means overprediction. Opposite signed errors can cancel. Empty groups are unavailable.')
        table(residuals.drop(columns=['Direction']))
        download(tables['residual_metrics'],'Download hourly residual summaries','residual_metrics.csv')
