import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

### ---------- 
### load datasets
import kagglehub

# Download latest version
filepath = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")
#filepath = './Data'


df_energy = pd.read_csv(
    os.path.join(filepath, 'energy_dataset.csv'), 
    parse_dates=['time']
)
df_energy['time'] =pd.to_datetime(df_energy['time'], utc=True) #, infer_datetime_format=True)

# df_weather = pd.read_csv(
#     os.path.join(filepath, 'weather_features.csv'), 
#     parse_dates=['dt_iso']
# )
# df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True) #, infer_datetime_format=True)
# df_weather['temp_C'] = df_weather.temp - 273 
### ----------

Freq_option = st.radio(
    "Show the Generation in frequency",
    ("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")
)

def AggDatabyFreq(freq):
    if freq == "Hourly":
        return df_energy
    else:
        freq_dict = {
            "Yearly":'YE',
            "Quarterly":'QE',
            "Monthly":'ME',
            "Weekly":'W',
            "Daily":'D',
        }
        return df_energy.groupby(pd.Grouper(key='time', axis=0,freq=freq_dict[freq])).sum().reset_index()

df_energy_agg = AggDatabyFreq(Freq_option)


tab1, tab2 = st.tabs(["Streamlit chart (default)", "Plotly chart"])
with tab1:
    st.subheader("Using Streamlit chart")
    st.title(f"Total {Freq_option} generation in Spain")
    st.line_chart(data = df_energy_agg,
                  x = 'time', y = ['generation solar','generation fossil gas'],
                  y_label = "MW")
with tab2:
    st.subheader("Using Plotly chart")
    fig = go.Figure()
    fig.update_layout(title=f"Total {Freq_option} generation in Spain",
                      yaxis=dict(title=dict(text="MW")))
    for gen in ['generation solar','generation fossil gas']:
        fig.add_trace(go.Scatter(x=df_energy_agg['time'], 
                                y=df_energy_agg[gen],
                        mode='lines',
                        name=gen))
    st.plotly_chart(fig, theme=None)










