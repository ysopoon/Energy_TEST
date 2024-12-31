import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np

# LLM agent
from langchain.agents import AgentType
#from langchain.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

def main():
    st.set_page_config(
        page_title="ChatBot",
        #page_icon="👋",
    )
    
    ### -------------------------------------------------- 
    df_energy = st.session_state['df_energy']
    df_weather = st.session_state['df_weather']
    ### --------------------------------------------------

    st.title("🦜 LangChain: Chat with Pandas DataFrame")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
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
            [df_energy,df_weather],
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