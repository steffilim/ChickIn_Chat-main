# base reference
import os
import re
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") 

# loading data
csv_file_path = 'data/bigDataHoliday.csv'
data = pd.read_csv(csv_file_path)

# initialising model
model = GoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.5
    )

# initialising dataframe agent
df_agent = create_pandas_dataframe_agent(model, data, verbose = True)

# setting keywords
keywords = re.compile(r'\b(selling price|province|unit|year)\b', re.IGNORECASE)

# app config
st.set_page_config(page_title="Conversational Chat", page_icon="🐔")
st.title("ChickIn Stream")

# functions

## response function from the bot
def get_response(user_query, chat_history):

    # condition statement
    if contains_keyword(user_query, keywords):
        # querying the dataset
        data_query = df_agent.run(user_query)
        data_reply = data_response(user_query, data_query, chat_history)
        return data_reply
    
    else: 
        return general_response(user_query, chat_history)

    
## for response not relating to data
def general_response(user_query, chat_history):
    template = """
    You are a helpful personal assistant for the poultry farmers in Indonesia.
    You are to answer their queries in the most helpful way possible.
    Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
        
    chain = prompt | model | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

## for response relating to data
def data_response(user_query, data_query, chat_history):
    template = """
    You are a helpful personal assistant for the chicken poultry farmers in Indonesia. 
    You are to assist them by answering their questions related to the chicken poultry farming industry.
    Answer the following questions considering the history of the conversation and the data queried:

    Chat history: {chat_history}

    User question: {user_question}

    Data queried: {data_query}

    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model | StrOutputParser()
    return chain.stream({"chat_history": chat_history, 
                                  "user_question": user_query,
                                  "data_query": data_query})

def contains_keyword(query, keywords):
    return keywords.search(query) is not None

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am ChickIn Chat. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))