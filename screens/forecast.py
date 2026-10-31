import streamlit as st
from forecasting import forecast_hour, verify_saved
from screens.common import show_forecast


def render(ctx):
    st.header('Forecast one hour')
    st.write('Replay a forecast issued at the start of an observed Test hour. Completed earlier hours supply history; the target outcome is revealed separately.')
    observed, bundle = ctx['observations'], ctx['bundle']
    hours = observed.loc[observed.period.eq('Test'), 'timestamp'].tolist()
    names = ctx['config']['model_order']
    hour_col, model_col = st.columns([3,2])
    hour = hour_col.selectbox('Select a test hour', hours, format_func=lambda t: t.strftime('%Y-%m-%d %H:%M'))
    name = model_col.selectbox('Select a model', names, index=names.index(bundle['default']))
    selection = (hour, name, bundle['run_id'])
    if st.session_state.get('forecast_selection') != selection:
        st.session_state.forecast_selection = selection
        st.session_state.pop('forecast_result', None)
        st.session_state.forecast_revealed = False
    if st.button('Forecast', type='primary'):
        try:
            result = forecast_hour(observed, bundle, hour, name)
            verify_saved(result, ctx['tables']['predictions'])
            st.session_state.forecast_result = result
            st.session_state.forecast_revealed = False
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get('forecast_result')
    if result is not None:
        if st.button('Reveal actual count'):
            st.session_state.forecast_revealed = True
        show_forecast(result, observed, st.session_state.get('forecast_revealed', False),ctx['tables']['interval_metrics'])
    else:
        st.info('Press Forecast to construct the inputs and apply the saved model.')
