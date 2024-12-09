import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title('DEMO -- Energy Generation and Wheather in Spain')
st.subheader('testing')

st.write("# Welcome to our app demo! 👋")


st.sidebar.success("Select a demo above.")

st.markdown(
    """
    testing using dataset in https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/data
"""
)
