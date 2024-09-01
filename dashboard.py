# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid_indices], y_pred[valid_indices]
    # Handle zero values by excluding them from the calculation
    nonzero_elements = y_true != 0
    return np.mean(np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements]))

# Sidebar with a link
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: left;">
            <img src="https://avatars.githubusercontent.com/u/4228249?v=4" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2>Created by Juan C Basurto</h2>", unsafe_allow_html=True)
    #

    st.markdown(
        """
        <div style="text-align: left;">
            <a href="https://github.com/jbasurtod" target="_blank"><p style="font-size:18px;"><img src="https://raw.githubusercontent.com/jbasurtod/streamlit_madrid_temperature_prediction/main/img/github-logo.png" width="25"> GithHub Profile</p></a>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Provide a brief summary with a link
    st.markdown("<h2>About This Streamlit App</h2>", unsafe_allow_html=True)
    st.write("""
    This visualization features productivized two XGBoost and LSTM machine learning models trained on temperature data from Madrid's open data portal. Using Airflow, Python and Google Drive, new temperature predictions are generated every hour as new data becomes available from the Ayuntamiento de Madrid [Open Data Portal](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=4985724ec1acd610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default). Its predictions are continuously compared with historical temperature records to provide pick the XGB/LSTM model used in the final productivized Temp Forecast App, which can be checked [through the following this link](https://bit.ly/madridtemp).
    """)


    st.markdown("<p style='font-size:12px;'>Origen de los datos en tiempo real: Ayuntamiento de Madrid. This app is not related in any way with Ayuntamiento de Madrid.</p>", 
        unsafe_allow_html=True)

# Set the title of the Streamlit page
st.markdown("<h1 style='margin-top: 0;'>Predicting Temperatures in Barrio San Isidro, Madrid</h1>", unsafe_allow_html=True)



# URLs for the CSV files
historic_url = 'https://drive.google.com/uc?id=12IUAsaeNNcNbIgXsXcR0UBqrKPx-ZncK'
predictions_url = 'https://drive.google.com/uc?id=1KXjcZsmCR5DMFXAAFQF34mhGtmnQI9qO'
predictions_url = 'https://drive.google.com/uc?id=1EofTofvRwQylkT_e8C0L5Iy0yUVO46dz'

#@st.cache_data
def load_data(url):
    return pd.read_csv(url)

# Load the dataframes
historic_df = load_data(historic_url)
predictions_df = load_data(predictions_url)

# Convert 'datetime' columns to datetime objects
historic_df['datetime'] = pd.to_datetime(historic_df['datetime'])
predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])

historic_df = historic_df.tail(30)

# Align the start date of predictions_df with the start date of historic_df
start_date = historic_df['datetime'].min()
predictions_df_filtered = predictions_df[predictions_df['datetime'] >= start_date]

# Merge the dataframes, ensuring predictions extend into the future
df_merged = pd.merge(historic_df, predictions_df_filtered, on='datetime', how='outer')

# Sort by datetime to ensure proper plotting order
df_merged.sort_values(by='datetime', inplace=True)

lstm_pred_column = 'pred_lstm'
xgb_pred_column = 'pred_xgb'

# Calculate MAPE for the last 30 hours
mape_xgb = mean_absolute_percentage_error(df_merged['temperature'], df_merged[xgb_pred_column])
mape_lstm = mean_absolute_percentage_error(df_merged['temperature'], df_merged[lstm_pred_column])

# Display the MAPE values side by side

st.markdown(f"""
    <div style="display: flex; justify-content: space-around;">
        <div style="color: darkorange; font-size: 25px; font-weight: bold;">
            24h XGB MAPE: {mape_xgb:.2%}
        </div>
        <div style="color: #5bcf6e; font-size: 25px; font-weight: bold;">
            24h LSTM MAPE: {mape_lstm:.2%}
        </div>
    </div>
""", unsafe_allow_html=True)


# Determine the last date of historic_df and add 30 minutes
last_historic_date = historic_df['datetime'].max()
last_update_time = last_historic_date + timedelta(minutes=30)
last_update_time_str = last_update_time.strftime('%Y-%m-%d %H:%M:%S')

# Define custom colors
historic_color = 'cadetblue'
xgb_color = 'darkorange'
lstm_color = '#5bcf6e'

# Create the Plotly figure
fig = go.Figure()

# Add traces for historic temperature, XGBoost predictions, and LSTM predictions
fig.add_trace(go.Scatter(
    x=df_merged['datetime'],
    y=df_merged['temperature'],
    mode='lines+markers',
    name='Historic Temperature',
    line=dict(color=historic_color, width=2),
    marker=dict(size=5)
))
fig.add_trace(go.Scatter(
    x=df_merged['datetime'],
    y=df_merged[xgb_pred_column],
    mode='lines',
    name='XGBoost Predictions',
    line=dict(color=xgb_color, width=2, dash='dash')
))
fig.add_trace(go.Scatter(
    x=df_merged['datetime'],
    y=df_merged[lstm_pred_column],
    mode='lines',
    name='LSTM Predictions',
    line=dict(color=lstm_color, width=2, dash='dash')
))

# Add small labels (annotations) every 3 hours and for last temperature point
last_label_time = None
last_temp_row = df_merged.dropna(subset=['temperature']).iloc[-1]

for i, row in df_merged.iterrows():
    # Annotate every 3 hours
    if last_label_time is None or (row['datetime'] - last_label_time).total_seconds() >= 3 * 3600:
        if not pd.isna(row['temperature']):
            fig.add_annotation(
                x=row['datetime'],
                y=row['temperature'],
                text=f"{row['temperature']:.2f}",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=historic_color)
            )
        if not pd.isna(row[xgb_pred_column]):
            fig.add_annotation(
                x=row['datetime'],
                y=row[xgb_pred_column],
                text=f"{row[xgb_pred_column]:.2f}",
                showarrow=False,
                yshift=-10,
                font=dict(size=10, color=xgb_color)
            )
        if not pd.isna(row[lstm_pred_column]):
            fig.add_annotation(
                x=row['datetime'],
                y=row[lstm_pred_column],
                text=f"{row[lstm_pred_column]:.2f}",
                showarrow=False,
                yshift=-10,
                font=dict(size=10, color=lstm_color)
            )
        last_label_time = row['datetime']

# Annotate the last 'temperature' data point and corresponding prediction points for xgb and lstm
last_temp_time = last_temp_row['datetime']
if not pd.isna(last_temp_row['temperature']):
    fig.add_annotation(
        x=last_temp_time,
        y=last_temp_row['temperature'],
        text=f"{last_temp_row['temperature']:.2f}",
        showarrow=False,
        yshift=10,
        font=dict(size=10, color=historic_color)
    )
if not pd.isna(last_temp_row[xgb_pred_column]):
    fig.add_annotation(
        x=last_temp_time,
        y=last_temp_row[xgb_pred_column],
        text=f"{last_temp_row[xgb_pred_column]:.2f}",
        showarrow=False,
        yshift=-10,
        font=dict(size=10, color=xgb_color)
    )
if not pd.isna(last_temp_row[lstm_pred_column]):
    fig.add_annotation(
        x=last_temp_time,
        y=last_temp_row[lstm_pred_column],
        text=f"{last_temp_row[lstm_pred_column]:.2f}",
        showarrow=False,
        yshift=-10,
        font=dict(size=10, color=lstm_color)
    )

# Update layout for the plot
fig.update_layout(
    width=2000,
    height=800,
    title=f"Actual vs Predicted Temperature (Last Update: {last_update_time_str})",
    xaxis_title="Date",
    yaxis_title="Temperature (Celsius)",
    xaxis=dict(title_font=dict(size=14), tickfont=dict(size=10)),
    yaxis=dict(title_font=dict(size=14), tickfont=dict(size=10)),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    )
)

# Show the plot in Streamlit
st.plotly_chart(fig)

# Filter the DataFrame to include only rows where 'temperature' is not NaN
table_df = df_merged[['datetime', 'temperature', xgb_pred_column, lstm_pred_column]].dropna(subset=['temperature'])

# Display the table in Streamlit
st.write("### Temperature and Predictions")
st.dataframe(table_df.rename(columns={'pred_xgb':'XGB Predictions','pred_lstm':'LSTM Predictions'}).round(1))
