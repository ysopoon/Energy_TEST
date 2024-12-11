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


## python packages
you may need to run the following to install all neccessary packages before running the app. 
```
# to run the streamlit app
pip install streamlit

# to load the data from Kaggle
pip install kagglehub

# to run the chatbot 
pip install tabulate
pip install langchain
pip install langchain-experimental
pip install openai

# to generate the plots
pip install plotly
```