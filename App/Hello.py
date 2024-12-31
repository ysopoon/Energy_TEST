import os
import pandas as pd
import streamlit as st


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title('DEMO -- Energy Generation')
    st.subheader('using entsoe data')

    st.write("# Welcome to our app demo! 👋")

    st.markdown(
        """
        The Review page provide both visualization and chatbot feature to review the energy 
        data in 5 european countries using data collected from https://transparency.entsoe.eu/.
    """
    )


if __name__ == "__main__":
    main()