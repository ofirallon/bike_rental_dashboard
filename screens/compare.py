import altair as alt
import pandas as pd
import streamlit as st
from screens.common import download
from screens.presentation import TEAL, BLUE, table, scope, number, chart_style


def render(ctx):
    st.header('Compare models')
    tables, config = ctx['tables'], ctx['config']
    with st.container(border=True):
        scope('Validation selection','Full Validation · 1 May–31 July 2012',
              f"Default: {config['default_model']} · selected by lowest Validation MAE, then frozen before Test.")
        summary = tables['validation_results']
        summary = summary.loc[summary.experiment.eq('principal') & summary.selected].copy()
        table(summary[['Model','MAE','Evaluated_Hours']],default_model=config['default_model'])
        with st.expander('Selected settings and selection rules'):
            import json
            settings=[]
            for row in summary.itertuples():
                params=json.loads(row.Parameters)
                if not params:
                    settings.append({'Model':row.Model,'Parameter':'Training lookup','Value':'Hour / working-day mean'})
                for key,value in params.items():
                    label={'alpha':'Penalty (alpha)','alpha_prior':'Prior precision','sigma2_noise':'Noise variance','noise_multiplier':'Noise multiplier'}.get(key,key)
                    settings.append({'Model':row.Model,'Parameter':label,'Value':number(value,4 if key in ['alpha','alpha_prior'] else 2)})
            table(pd.DataFrame(settings),default_model=config['default_model'])
            st.caption('Exact candidate ties retain the first tested setting. Exact method ties use: '+', '.join(config['model_order'])+'.')
    metrics = tables['metrics']
    with st.container(border=True):
        scope('Full Test scores','Full Test · 1 Aug–31 Dec 2012','Fixed scores across every observed Test hour. Date filters below affect only the forecast plot and its export.')
        plot=metrics.melt(id_vars=['Model','Evaluated_Hours'],value_vars=['MAE','RMSE'],var_name='Metric',value_name='Error')
        plot['Model label']=plot.Model.map(lambda n:n+' · Default' if n==config['default_model'] else n)
        order=[n+' · Default' if n==config['default_model'] else n for n in config['model_order']]
        bars=alt.Chart(plot).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X('Error:Q',title='Point error (rentals per hour) · lower is better'),
            y=alt.Y('Model label:N',sort=order,title=None),yOffset='Metric:N',
            color=alt.Color('Metric:N',scale=alt.Scale(domain=['MAE','RMSE'],range=[TEAL,BLUE]),title=None),
            tooltip=['Model:N','Metric:N',alt.Tooltip('Error:Q',format=',.2f'),alt.Tooltip('Evaluated_Hours:Q',title='Evaluated hours',format=',.0f')])
        st.altair_chart(chart_style(bars.properties(height=280)),width='stretch')
        table(metrics,formats={'Training_Time_sec':4},default_model=config['default_model'])
        st.caption(config['training_time_definition']+'. MAE and RMSE are rentals per hour; lower is better. Display rounding does not change saved results.')
        download(metrics, 'Download complete score table', 'metrics.csv')
        p = metrics.loc[metrics.Model.eq('Poisson')].iloc[0]
        st.metric('Poisson mean negative log count probability (natural log)',number(p.Poisson_NLL))
        st.caption(f'Full Test n={int(p.Evaluated_Hours):,}; computed with -poisson.logpmf(actual, fitted mean). This is a discrete count-probability score.')
    scope('Actual versus forecast','Selected range',selected=True)
    model_col,date_col=st.columns([2,3])
    name = model_col.selectbox('Model to plot', config['model_order'], index=config['model_order'].index(config['default_model']))
    dates = date_col.date_input('Test plot date range',value=(pd.Timestamp('2012-08-01').date(),pd.Timestamp('2012-08-10').date()),min_value=pd.Timestamp('2012-08-01').date(),max_value=pd.Timestamp('2012-12-31').date())
    if len(dates)!=2:
        st.info('Select both dates.')
        return
    start,end = pd.Timestamp(dates[0]),pd.Timestamp(dates[1])+pd.Timedelta(hours=23)
    predictions=tables['predictions']
    rows=predictions.loc[predictions.period.eq('Test') & predictions.model.eq(name) & pd.to_datetime(predictions.timestamp).between(start,end)].copy()
    scope(name,f'{start:%d %b %Y} – {end:%d %b %Y}',f'{len(rows):,} observed Test hours · applies to this plot and its CSV only.',selected=True)
    if rows.empty:
        st.info('No observed target hours in this range.')
        return
    rows['timestamp']=pd.to_datetime(rows.timestamp)
    chart=rows.set_index('timestamp')[['cnt','scored_forecast']].reindex(pd.date_range(start,end,freq='h'))
    chart['segment']=chart.cnt.isna().cumsum().astype(str)
    chart=chart.rename_axis('timestamp').reset_index().melt(id_vars=['timestamp','segment'],value_vars=['cnt','scored_forecast'],var_name='series',value_name='rentals')
    chart['series']=chart.series.map({'cnt':'Recorded actual','scored_forecast':'Forecast'})
    lines=alt.Chart(chart.dropna(subset=['rentals'])).mark_line().encode(x=alt.X('timestamp:T',title='Hour'),y=alt.Y('rentals:Q',title='Rentals per hour'),
        color=alt.Color('series:N',scale=alt.Scale(domain=['Recorded actual','Forecast'],range=[BLUE,TEAL]),title=None),
        strokeDash=alt.StrokeDash('series:N',title=None),detail='segment:N',
        tooltip=[alt.Tooltip('timestamp:T',title='Hour'),'series:N',alt.Tooltip('rentals:Q',format=',.2f')])
    st.altair_chart(chart_style(lines),width='stretch')
    download(rows,'Download displayed forecasts','displayed_forecasts.csv')
