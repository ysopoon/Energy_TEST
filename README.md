# Energy_TEST

## introduction
To run the streamlit app
1. make sure you have all python packages installed
2. clone / download this project to your local
3. run the following in command prompt (Win) or Terminal (Mac)
```
streamlit run <project folder>/App/Hello.py
```

**Note**: If you want to use the chatbot, you will need to obtain your own OpenAI API Key. Make sure you have credit balance in your OpenAI account, else you will get error when you play with the chatbot. 

**Note 2**: 
- There is Chatbot in the **Review** page. This bot uses key saved to ***.streamlit/secrets.toml*** or *App/API_keys/OpenAI_API_Keys.txt*. Make sure you save your API key in the file before running the app. 



## python packages
you may need to run the following to install all neccessary packages before running the app. 
```
# to run the streamlit app
pip install streamlit

# to run the chatbot 
pip install tabulate
pip install langchain
pip install langchain-experimental
## pip install -U langchain-community
pip install openai
pip install langchain_openai
## pip install langchain-core langgraph>0.2.27
pip install langchain_google_genai

# to generate the plots
pip install plotly
```