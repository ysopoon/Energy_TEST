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
from langchain.chat_models import ChatOpenAI
#from langchain-community.chat_models import ChatOpenAI


@st.cache_data 
def read_entsoe_df(country):
    filepath = os.path.join(os.path.dirname( __file__ ),'../../Data')
    table = pd.DataFrame()
    for i in [2022, 2023]:
        df = pd.read_csv(os.path.join(filepath, f'energy_{i}_{country}.csv'))

        df = df.replace('n/e', np.nan)
        df[[x for x in df.columns if "MW" in x ]] = \
            df[[x for x in df.columns if "MW" in x ]] \
                .apply(lambda x: x.astype(float), axis=1) \
        #         .interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
        df.rename(columns=lambda x: x.replace(' - Actual Aggregated [MW]', ' [MW]'), inplace=True)

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
    filepath = os.path.join(os.path.dirname( __file__ ),'../API_keys')
    with open(os.path.join(filepath, 'OpenAI_API_Keys.txt'), 'r') as file:
        api_key = file.readlines()
    return api_key[0]

def get_country_name(abb):
    country_dict = {
        "FR":'France',
        "DE":'Germany',
        "IT":'Italy',
        "PT":'Portugal',
        "ES":'Spain',
    }
    return country_dict[abb]

def main():
    st.set_page_config(
        page_title="Review entsoe data",
        #page_icon="👋",
    )

    st.title("Basic Analysis with Visualization using entsoe data")
    st.write("using data from https://transparency.entsoe.eu/")
    st.write("select the countries you want to review on the left.")

    display = st.sidebar.radio("Select", ["Dashboard", "Chatbot"])

    st.sidebar.success("Select the countries")
    FR = st.sidebar.checkbox("France", key = "selected_FR")
    DE = st.sidebar.checkbox("Germany", key = "selected_DE")
    IT = st.sidebar.checkbox("Italy", key = "selected_IT")
    PT = st.sidebar.checkbox("Portugal", key = "selected_PT")
    ES = st.sidebar.checkbox("Spain", value = True, key = "selected_ES")

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

    if display == "Dashboard":
        dashboard()
    else:
        chatbot()



def dashboard():
    countries = ['FR','DE','IT','PT','ES']
    selected_countries = [country for country in countries if st.session_state["selected_"+country]]
    
    st.header("Actual Generation per Production Type")

    col1, col2 = st.columns(2)
    with col1:
        d = st.date_input("Review the generation as of ", 
                        value = datetime.date(2023, 4, 30), 
                        min_value = datetime.date(2022, 1, 1),
                        max_value = datetime.date(2023, 12, 31),)
    with col2:
        chart_type = st.radio("Select the chart type", ['bar', 'heatmap'], horizontal=True)

    for tab, country in zip(st.tabs(selected_countries), selected_countries):
        with tab:
            st.header(get_country_name(country))
            df = st.session_state["df_energy_"+country]
            df = df[df['date'] == d]
            if chart_type == 'bar':
                fig = px.bar(df, 
                            x="time", 
                            y=[x for x in df.columns if "MW" in x ]
                            )
            elif chart_type == 'heatmap':
                fig = px.imshow(img= df[[x for x in df.columns if "MW" in x ]].T,
                                x = df['time'],
                                y = [x for x in df.columns if "MW" in x ])
            st.plotly_chart(fig)
    

    Freq_option = st.radio(
        "Show the Generation in frequency",
        ("Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"), 
        horizontal=True
    )

    @st.cache_data
    def AggDatabyFreq(df, freq):
        if freq == "Hourly":
            return df.copy()
        else:
            freq_dict = {
                "Yearly":'YE',
                "Quarterly":'QE',
                "Monthly":'ME',
                "Weekly":'W',
                "Daily":'D',
            }
            df1 = df.copy().drop('date', axis=1)
            return df1.groupby(pd.Grouper(key='time', axis=0,freq=freq_dict[freq])).sum().reset_index()

    selected_gen = st.selectbox(
            "Select the energy",
            st.session_state['energy_gen'],
        )

    fig = go.Figure()
    fig.update_layout(title=f"Total {Freq_option} {selected_gen.replace('[MW]', '')} generation",
                    yaxis=dict(title=dict(text="MW")))
    for country in countries:
        if st.session_state["selected_"+country]:
            df_agg = AggDatabyFreq(st.session_state["df_energy_"+country], Freq_option)
            for gen in [selected_gen]:
                fig.add_trace(go.Scatter(x=df_agg['time'], 
                                        y=df_agg[gen],
                                mode='lines',
                                name=get_country_name(country)))
    if Freq_option not in ['Yearly', 'Quaterly']:
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),type="date" ))
    st.plotly_chart(fig, theme=None)




def chatbot():
    st.title("🦜 LangChain: Chat with Pandas DataFrame")

    all_countries_df = []
    for country in ['FR','DE','IT','PT','ES']:
        if st.session_state["selected_"+country]:
            all_countries_df.append(st.session_state["df_energy_"+country])
            
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except:
        openai_api_key = get_openai_api_key()

    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(
            temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            all_countries_df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
        )    

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)


if __name__ == "__main__":
    main()