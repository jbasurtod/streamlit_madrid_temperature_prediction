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
    st.markdown("[GitHub Profile](https://github.com/jbasurtod)", unsafe_allow_html=True)

# Set the title of the Streamlit page
st.markdown("<h1 style='margin-top: 0;'>Predicting Temperatures in Madrid - Station 'Farolillo'</h1>", unsafe_allow_html=True)

# Provide a brief summary with a link
st.write("""
This visualization features a productivized XGBoost machine learning model trained on temperature data from Madrid's open data portal. The model updates hourly using real-time temperature data [available here](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=4985724ec1acd610VgnVCM1000001d4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default). Its predictions are continuously compared with historical temperature records to provide insights into trends and forecast accuracy. This productivized project uses Airflow, Google Drive and Streamlit.
""")

# URLs for the CSV files
historic_url = 'https://drive.google.com/uc?id=12IUAsaeNNcNbIgXsXcR0UBqrKPx-ZncK'
predictions_url = 'https://drive.google.com/uc?id=1KXjcZsmCR5DMFXAAFQF34mhGtmnQI9qO'

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

# Calculate MAPE for the last 30 hours
mape_value = mean_absolute_percentage_error(df_merged['temperature'], df_merged['predictions'])

# Display the MAPE value with larger font, bold, and pastel blue color
st.markdown(f"<h4 style='color: #5b95cf;'><strong>Last 30 hours MAPE: {mape_value:.2%}</strong></h4>", unsafe_allow_html=True)

# Determine the last date of historic_df and add 30 minutes
last_historic_date = historic_df['datetime'].max()
last_update_time = last_historic_date + timedelta(minutes=30)
last_update_time_str = last_update_time.strftime('%Y-%m-%d %H:%M:%S')

# Define custom colors
historic_color = 'cadetblue'
predictions_color = 'darkorange'

# Create figure using Plotly graph objects
fig = go.Figure()

# Add historic_df line (solid)
fig.add_trace(go.Scatter(
    x=df_merged['datetime'],
    y=df_merged['temperature'],
    mode='lines+markers',
    name='Historic Temperature',
    line=dict(color=historic_color, width=2),
    marker=dict(size=5)
))

# Add predictions_df line (dashed)
fig.add_trace(go.Scatter(
    x=df_merged['datetime'],
    y=df_merged['predictions'],
    mode='lines',  # Removed markers
    name='Predicted Temperature',
    line=dict(color=predictions_color, width=2, dash='dash')  # Changed to dashed line
))

# Add small labels (annotations) to specific points in historic_df only
for i, row in df_merged.iterrows():
    if not pd.isna(row['temperature']):
        fig.add_annotation(
            x=row['datetime'],
            y=row['temperature'],
            text=f"{row['temperature']:.2f}",
            showarrow=False,
            yshift=10,
            font=dict(size=10, color=historic_color)
        )
    if not pd.isna(row['predictions']) and (row['datetime'] <= last_historic_date or i % 3 == 0):
        fig.add_annotation(
            x=row['datetime'],
            y=row['predictions'],
            text=f"{row['predictions']:.2f}",
            showarrow=False,
            yshift=-10,
            font=dict(size=10, color=predictions_color)
        )

# Update layout for the plot
fig.update_layout(
    width=2000,
    height=800,
    title=f"Actual vs Predicted Temperature (Last Update: {last_update_time_str})",
    xaxis_title="Date",
    yaxis_title="Temperature",
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
