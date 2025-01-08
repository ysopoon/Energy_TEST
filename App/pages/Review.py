# Importing necessary libraries
import os
import datetime
#from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LLM agent
from langchain.agents import AgentType
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler


from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
#from langchain-community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_core.messages import AIMessage, HumanMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory



@st.cache_data 
def read_entsoe_df(country):
    # Load energy data for the specified country from CSV files
    filepath = os.path.join(os.path.dirname( __file__ ),'../../Data')
    table = pd.DataFrame()
    for i in [2022, 2023]:
        df = pd.read_csv(os.path.join(filepath, f'energy_{i}_{country}.csv'))

        # Replace 'n/e' with NaN and convert columns with "MW" to float
        df = df.replace('n/e', np.nan)
        df[[x for x in df.columns if "MW" in x ]] = \
            df[[x for x in df.columns if "MW" in x ]] \
                .apply(lambda x: x.astype(float), axis=1) \
        #         .interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
        df.rename(columns=lambda x: x.replace(' - Actual Aggregated [MW]', ' [MW]'), inplace=True)

        # Parse date and time columns
        df[['start_time', 'time']] = df['MTU'].str.split(' - ', expand=True)
        df['time'] = df['time'].apply(lambda x: x.replace(' (UTC)','')) \
                                .apply(lambda x: datetime.datetime.strptime(x, "%d.%m.%Y %H:%M"))
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df['date'] = df['time'].dt.date
        df = df[[x for x in df.columns if x not in ['MTU','start_time']]]
        table = df if table.empty else pd.concat([table,df])
    return table

@st.cache_data
def get_openai_api_key():
    # Retrieve the OpenAI API key from a local file
    filepath = os.path.join(os.path.dirname( __file__ ),'../API_keys')
    with open(os.path.join(filepath, 'OpenAI_API_Keys.txt'), 'r') as file:
        api_key = file.readlines()
    return api_key[0]

def get_country_name(abb):
    # Map country abbreviations to full names
    country_dict = {
        "FR":'France',
        "DE":'Germany',
        "IT":'Italy',
        "PT":'Portugal',
        "ES":'Spain',
    }
    return country_dict[abb]

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Review entsoe data",
        #page_icon="👋",
    )

    st.title("Basic Analysis with Visualization using entsoe data")
    st.write("using data from https://transparency.entsoe.eu/")
    st.write("select the countries you want to review on the left.")

    # Sidebar options for dashboard or chatbot
    display = st.sidebar.radio("Select", ["Dashboard", "Chatbot"])

    st.sidebar.text("Select the countries")
    FR = st.sidebar.checkbox("France", key = "selected_FR")
    DE = st.sidebar.checkbox("Germany", key = "selected_DE")
    IT = st.sidebar.checkbox("Italy", key = "selected_IT")
    PT = st.sidebar.checkbox("Portugal", key = "selected_PT")
    ES = st.sidebar.checkbox("Spain", value = True, key = "selected_ES")

    # Load data into session state for selected countries
    if FR and "df_energy_FR" not in st.session_state:
        df_energy_FR = read_entsoe_df('FR')
        st.session_state['df_energy_FR'] = df_energy_FR
    if DE and "df_energy_DE" not in st.session_state:
        df_energy_DE = read_entsoe_df('DE')
        st.session_state['df_energy_DE'] = df_energy_DE
    if IT and "df_energy_IT" not in st.session_state:
        df_energy_IT = read_entsoe_df('IT')
        st.session_state['df_energy_IT'] = df_energy_IT
    if PT and "df_energy_PT" not in st.session_state:
        df_energy_PT = read_entsoe_df('PT')
        st.session_state['df_energy_PT'] = df_energy_PT
    if ES and "df_energy_ES" not in st.session_state:
        df_energy_ES = read_entsoe_df('ES')
        st.session_state['df_energy_ES'] = df_energy_ES
        st.session_state['energy_gen'] = [x for x in df_energy_ES.columns if "MW" in x ]

    # Validate if at least one country is selected
    if not (FR or DE or IT or PT or ES):
        # Show a warning message if no countries are selected
        st.warning("Please select at least one country to display the data.")
        return

    # Display selected section (Dashboard or Chatbot)
    if display == "Dashboard":
        dashboard()
    else:
        chatbot()



def dashboard():
    countries = ['FR', 'DE', 'IT', 'PT', 'ES']
    
    # Get selected countries
    selected_countries = [country for country in countries if st.session_state["selected_"+country]]
    
    st.header("Actual Generation per Production Type")
    
    # Create widgets for date selection and chart type
    col1, col2 = st.columns(2)
    with col1:
        d = st.date_input(
            "Review the generation as of ",
            value=datetime.date(2023, 4, 30),
            min_value=datetime.date(2022, 1, 1),
            max_value=datetime.date(2023, 12, 31),
        )
    with col2:
        # Chart type selection
        chart_type = st.radio(
            "Select the chart type",
            ["bar", "heatmap"],
            horizontal=True
        )

    # Create tabs for selected countries
    for tab, country in zip(st.tabs(selected_countries), selected_countries):
        with tab:
            # Show country name as header in each tab
            st.header(get_country_name(country))
            
            # Filter DataFrame by selected date
            df = st.session_state["df_energy_"+country]
            df = df[df['date'] == d]

            # Display data in the selected chart type (bar or heatmap)
            if chart_type == 'bar':
                fig = px.bar(
                    df,
                    x="time",
                    y=[x for x in df.columns if "MW" in x]
                )
            elif chart_type == 'heatmap':
                fig = px.imshow(
                    img=df[[x for x in df.columns if "MW" in x]].T,
                    x=df['time'],
                    y=[x for x in df.columns if "MW" in x]
                )
            # Display the chart in the corresponding tab
            st.plotly_chart(fig)
    

    st.header("Compare Actual Generation by Production Type and Frequency")
    # Frequency options for data aggregation
    Freq_option = st.radio(
        "Show the Generation in frequency",
        ("Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"), 
        horizontal=True,
    )

    @st.cache_data
    def AggDatabyFreq(df, freq):
        freq_dict = {
            "Yearly":'YE',
            "Quarterly":'QE',
            "Monthly":'ME',
            "Weekly":'W',
            "Daily":'D',
            "Hourly":'h',
        }
        df1 = df.copy().drop('date', axis=1)
        return df1.groupby(pd.Grouper(key='time', axis=0,freq=freq_dict[freq])).sum().reset_index()

    # Dropdown to select energy type
    selected_gen = st.selectbox(
            "Select the energy",
            st.session_state['energy_gen'],
        )

    # Create and display the aggregated data chart
    fig1 = go.Figure()
    fig1.update_layout(title=f"Total {Freq_option} {selected_gen.replace('[MW]', '')} generation",
                    yaxis=dict(title=dict(text="MW")))
    for country in selected_countries:
        df_agg = AggDatabyFreq(st.session_state["df_energy_"+country], Freq_option)
        for gen in [selected_gen]:
            fig1.add_trace(go.Scatter(x=df_agg['time'], 
                                    y=df_agg[gen],
                            mode='lines',
                            name=get_country_name(country)))
    if Freq_option not in ['Yearly', 'Quaterly']:
        fig1.update_layout(xaxis=dict(rangeslider=dict(visible=True),type="date" ))
    st.plotly_chart(fig1, theme=None)


def chatbot():
    #st.title("🦜 LangChain: Chat with Pandas DataFrame") 
    st.write("Review the data using Chatbot")

    # create a list of data frame that we want to review
    all_countries_df = []
    for country in ['FR','DE','IT','PT','ES']:
        if st.session_state["selected_"+country]:
            all_countries_df.append(st.session_state["df_energy_"+country])

    LLM_option = st.selectbox(
        "What LLM model do you want to use?",
        ("OpenAI GPT", "Google Gemini"),
        label_visibility="hidden"
    )

    if LLM_option == "OpenAI GPT":
        # get the OpenAI API Key        
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]   
        except:
            try:
                openai_api_key = get_openai_api_key()
            except:
                st.info("Please add your OpenAI API key to continue.")
                st.stop() 
        llm = ChatOpenAI(
            temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True
        )
        agent_type=AgentType.OPENAI_FUNCTIONS
    elif LLM_option == "Google Gemini":
        # get the Google API Key        
        try:
            google_api_key = st.secrets["GOOGLE_API_KEY"]   
        except:
            st.info("Please add your Google API key to continue.")
            st.stop() 
        llm = ChatGoogleGenerativeAI(    
            google_api_key=google_api_key, model="gemini-1.5-pro"
        )
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    

    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", 
                                         #"content": "How can I help you?"}]
                                         "content": "What do you want to know about the energy generation in the selected countries?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            all_countries_df,
            verbose=True,
            agent_type=agent_type,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            suffix = """
            when filtering by the 'date' column, you should compare using datetime format.
            for example, if filter for Dec 1, 2023, do:
            `df['date'] == datetime.date(2023, 12, 1)`

            the 'Area' columns is in "country (country abb)" format. 
            for example, if it's Spain data, you will have "Spain (ES)".
            when filtering by the 'Area' column, e.g. filtering by Spain, do:
            `df['Area'].str.contains('Spain')`
            """
        )    

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            #output = pandas_df_agent.invoke({"input": prompt}, {"callbacks": [st_cb]})
            #response = output["output"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)



if __name__ == "__main__":
    main()