import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    st.set_page_config(
        page_title="Basic Analysis with Visualization",
        #page_icon="👋",
    )



    ### -------------------------------------------------- 
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

    df_weather = pd.read_csv(
        os.path.join(filepath, 'weather_features.csv'), 
        parse_dates=['dt_iso']
    )
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True) #, infer_datetime_format=True)
    # df_weather['temp_C'] = df_weather.temp - 273 

    # drop duplicate rows in df_weather
    df_weather = df_weather.drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time').reset_index()
    ### --------------------------------------------------





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




    st.subheader("Energy Generatoin and the weather")

    min_d = df_energy['time'].min()
    max_d = df_energy['time'].max()

    selected_date = st.date_input(
        "Select the date you want to review",
        (max_d, max_d),
        min_d,
        max_d,
        format="MM.DD.YYYY",
    )

    selected_gen = st.multiselect(
        "Select Energy Generation",
        [i for i in df_energy.columns if "generation" in i],
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_weather = st.selectbox(
            "Select the weather",
            ["temp","temp_min","temp_max","pressure","humidity","wind_speed",
            "wind_deg","rain_1h","rain_3h","snow_3h","clouds_all"],
        )

    with col2:
        selected_city = st.segmented_control(
            "in City", 
            df_weather.city_name.unique(), 
            selection_mode="multi"
        )

    def filter_df_by_date(df, date):
        df['date'] = df['time'].dt.date
        df_filtered = df[(df['date'] >= date[0]) & (df['date'] <= date[1])]
        return df_filtered


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_layout(title=f"")
    df_energy_filtered = filter_df_by_date(df_energy, selected_date)
    for gen in selected_gen:
        fig.add_trace(go.Scatter(
            x=df_energy_filtered['time'], 
            y=df_energy_filtered[gen],
            mode='lines',
            name=gen,
            legendgroup='Energy')
        )
    fig.update_yaxes(title_text="MW", secondary_y=False)
    df_weather_filtered = filter_df_by_date(df_weather, selected_date)
    for weather in [selected_weather]:
        for city in selected_city:
            fig.add_trace(go.Scatter(
                x=df_weather_filtered[df_weather_filtered['city_name'] == city]['time'], 
                y=df_weather_filtered[df_weather_filtered['city_name'] == city][weather],
                mode='lines+markers',
                name=f"{weather} {city}",
                legendgroup='Weather'
                ),
                secondary_y=True,
            )

    # Update layout to show the second legend
    fig.update_layout(
        legend=dict(traceorder='grouped'),
        legend2=dict(x=1,y=1,traceorder='grouped')
    )


    st.plotly_chart(fig, theme=None)


if __name__ == "__main__":
    main()