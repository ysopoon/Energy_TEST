import os
import pandas as pd
import streamlit as st


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title('DEMO -- Energy Generation and Wheather in Spain')
    st.subheader('testing')

    st.write("# Welcome to our app demo! 👋")

    load_data()

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        testing using dataset in https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/data
    """
    )



@st.cache_data
def load_data():
    ### load datasets
    # Download latest version
    import kagglehub
    filepath = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")

    ## To share the same dataframe between pages
    # Initialization session state
    if 'df_energy' not in st.session_state:
        df_energy = pd.read_csv(
            os.path.join(filepath, 'energy_dataset.csv'), 
            parse_dates=['time']
        )
        df_energy['time'] =pd.to_datetime(df_energy['time'], utc=True) #, infer_datetime_format=True)
        st.session_state['df_energy'] = df_energy
    
    if 'df_weather' not in st.session_state:
        df_weather = pd.read_csv(
            os.path.join(filepath, 'weather_features.csv'), 
            parse_dates=['dt_iso']
        )
        df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True) #, infer_datetime_format=True)
        # df_weather['temp_C'] = df_weather.temp - 273 

        # drop duplicate rows in df_weather
        df_weather = df_weather.drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time').reset_index()
        st.session_state['df_weather'] = df_weather
    ### --------------------------------------------------

if __name__ == "__main__":
    main()