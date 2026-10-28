import streamlit as st
import pandas as pd
import json

# Basic page setup
st.set_page_config(page_title="Bike-Rental Forecast Dashboard", layout="wide")

# Load data from the files we generated earlier
@st.cache_data
def load_data():
    df_audit = pd.read_csv("data_audit.csv")
    df_metrics = pd.read_csv("metrics.csv")
    df_preds = pd.read_csv("predictions.csv")
    df_val = pd.read_csv("validation_results.csv")
    
    with open("config.json", "r") as f:
        config = json.load(f)
        
    return df_audit, df_metrics, df_preds, df_val, config

df_audit, df_metrics, df_preds, df_val, config = load_data()

# Sidebar menu for navigating between the 4 required screens
st.sidebar.title("Navigation")
screen = st.sidebar.radio("Go to:", [
    "Data overview",
    "Forecast one hour",
    "Compare models",
    "Inspect model behaviour"
])

# Main app title
st.title("🚲 Bike-Rental Forecast Dashboard")


# Screen 1: Data overview
if screen == "Data overview":
    st.header("Data Overview")
    st.write("Browse observed hours, missing timestamps, and data quality findings.")
    
    st.subheader("Data Audit Results")
    st.dataframe(df_audit, width='stretch') # Fixed the container width warning
    
    st.subheader("Time Periods Configuration")
    st.json(config['periods'])
    
    st.info("👈 Use the sidebar to navigate to other screens.")


# Screen 2: Forecast one hour 
elif screen == "Forecast one hour":
    st.header("Forecast one hour")
    st.write("Replay a historical forecast. Choose a test hour and a model to see the prediction and then reveal the actual outcome.")
    
    # Selectors for Hour and Model
    test_preds = df_preds.copy()
    test_hours = test_preds['timestamp'].unique()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_hour = st.selectbox("Select a test hour:", test_hours)
    with col2:
        available_models = test_preds['model'].unique()
        selected_model = st.selectbox("Select a model:", available_models)
        
    # Get the specific prediction row
    pred_row = test_preds[(test_preds['timestamp'] == selected_hour) & (test_preds['model'] == selected_model)]
    
    if not pred_row.empty:
        pred_row = pred_row.iloc[0]
        
        st.divider()
        
        # Display Forecast Inputs and Prediction
        st.subheader(f"Forecast for {selected_hour} using {selected_model}")
        
        col_pred, col_intervals = st.columns(2)
        with col_pred:
            st.metric(label="Predicted Rental Count", value=f"{pred_row['scored_forecast']:.2f}")
            
        with col_intervals:
            if pd.notna(pred_row['lower_80']):
                st.write("**Prediction Intervals:**")
                st.write(f"- 80% Interval: [{pred_row['lower_80']:.2f}, {pred_row['upper_80']:.2f}]")
                st.write(f"- 95% Interval: [{pred_row['lower_95']:.2f}, {pred_row['upper_95']:.2f}]")
            else:
                st.write("**Prediction Intervals:**")
                st.write("Intervals unavailable for this method.")
                
        # Reveal Actual Count Button
        st.divider()
        
        # Use session state to manage the reveal button so it resets when changing selection
        if 'revealed' not in st.session_state:
            st.session_state.revealed = False
            
        if 'prev_selection' not in st.session_state:
            st.session_state.prev_selection = (selected_hour, selected_model)
        elif st.session_state.prev_selection != (selected_hour, selected_model):
            st.session_state.revealed = False
            st.session_state.prev_selection = (selected_hour, selected_model)

        if st.button("Reveal actual count"):
            st.session_state.revealed = True
            
        if st.session_state.revealed:
            st.success(f"**Recorded Actual Count: {pred_row['cnt']:.0f}**")
            st.write(f"**Forecast Error:** {pred_row['error']:.2f} (Absolute Error: {pred_row['abs_error']:.2f})")


# Screen 3: Compare models
elif screen == "Compare models":
    st.header("Compare models")
    st.write("Inspect test scores for all models and visualize forecasts against recorded counts.")
    
    # 1. Summary Table
    st.subheader("Test Period Performance (Full Test Set)")
    
    # Highlight the default model
    def highlight_default(row):
        if row['Model'] == config['default_model']:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
        
    st.dataframe(df_metrics.style.apply(highlight_default, axis=1), width='stretch')
    st.caption(f"Note: Green row indicates the Default Model ({config['default_model']}) selected during validation.")
    
    st.divider()
    
    # 2. Interactive Time Plot
    st.subheader("Forecast vs Actuals Plot")
    
    # Selectors for the plot
    available_models = df_preds['model'].unique()
    plot_model = st.selectbox("Select a model to plot:", available_models, key="plot_model")
    
    # Simple date range slider
    test_dates = pd.to_datetime(df_preds['timestamp']).dt.date.unique()
    start_date, end_date = st.select_slider(
        "Select Date Range:",
        options=test_dates,
        value=(test_dates[0], test_dates[10]) # Default to first 10 days
    )
    
    # Filter data for the plot
    df_preds['date'] = pd.to_datetime(df_preds['timestamp']).dt.date
    plot_data = df_preds[(df_preds['model'] == plot_model) & 
                         (df_preds['date'] >= start_date) & 
                         (df_preds['date'] <= end_date)]
                         
    if not plot_data.empty:
        st.line_chart(
            plot_data.set_index('timestamp')[['cnt', 'scored_forecast']],
            use_container_width=True
        )
    
    st.divider()
    
    # 3. CSV Downloads
    st.subheader("Download Results")
    csv_preds = plot_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download displayed forecasts as CSV",
        data=csv_preds,
        file_name='displayed_forecasts.csv',
        mime='text/csv',
    )

    # Screen 4: Inspect model behaviour
elif screen == "Inspect model behaviour":
    st.header("Inspect model behaviour")
    st.write("Analyze the default model's largest errors, residual patterns, and interval coverage.")
    
    default_model = config['default_model']
    
    st.divider()
    
    # 1. Largest Errors for Default Model
    st.subheader(f"Top 10 Largest Errors ({default_model})")
    df_default = df_preds[(df_preds['model'] == default_model) & (df_preds['period'] == 'Test')].copy()
    
    top_10_errors = df_default.nlargest(10, 'abs_error')
    display_cols = ['timestamp', 'cnt', 'scored_forecast', 'error', 'abs_error']
    st.dataframe(top_10_errors[display_cols], width='stretch')
    
    st.divider()
    
    # 2. Residuals by hour
    st.subheader(f"Residuals by Hour of Day ({default_model})")
    st.write("Positive values mean the model underpredicted; negative values mean it overpredicted.")
    df_default['hour'] = pd.to_datetime(df_default['timestamp']).dt.hour
    residuals_by_hour = df_default.groupby('hour')['error'].mean().reset_index()
    
    st.bar_chart(residuals_by_hour.set_index('hour')['error'], use_container_width=True)
    
    st.divider()
    
    # 3. Interval Coverage (For Poisson and Bayesian)
    st.subheader("Interval Coverage Audit")
    probabilistic_models = [m for m in df_preds['model'].unique() if m in ['Poisson', 'Bayesian']]
    
    if probabilistic_models:
        selected_prob_model = st.selectbox("Select model to audit coverage:", probabilistic_models)
        df_prob = df_preds[(df_preds['model'] == selected_prob_model) & (df_preds['period'] == 'Test')].copy()
        
        # Calculate coverage (1 if actual count is between lower and upper bounds, 0 otherwise)
        df_prob['covered_80'] = (df_prob['cnt'] >= df_prob['lower_80']) & (df_prob['cnt'] <= df_prob['upper_80'])
        df_prob['covered_95'] = (df_prob['cnt'] >= df_prob['lower_95']) & (df_prob['cnt'] <= df_prob['upper_95'])
        
        cov_80_pct = df_prob['covered_80'].mean() * 100
        cov_95_pct = df_prob['covered_95'].mean() * 100
        
        col_80, col_95 = st.columns(2)
        with col_80:
            st.metric(label="Observed 80% Coverage", value=f"{cov_80_pct:.1f}%", delta=f"{cov_80_pct - 80:.1f}% from target")
        with col_95:
            st.metric(label="Observed 95% Coverage", value=f"{cov_95_pct:.1f}%", delta=f"{cov_95_pct - 95:.1f}% from target")
    else:
        st.info("No probabilistic models available to calculate intervals.")