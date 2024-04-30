import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# langchain and gemini library imports
import os
import langchain
#from langchain import LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# agent library imports
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# AI Chat library imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from langchain.retrievers import DataFrameRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain


class DataFrameRetriever:
    def __init__(self, df, text_column):
        self.vectoriser = TfidfVectorizer()
        self.model = NearestNeighbors(n_neighbors = 5, algorithm='auto')
        self.df = df
        self.fit(df[text_column])

    def fit(self, text_data):
        tfidf_matrix = self.vectoriser.fit_transform(text_data)
        self.model.fit(tfidf_matrix)

    def search(self, query):
        query_tfidf = self.vectoriser.transform([query])
        distances, indices = self.model.kneighbors(query_tfidf)
        return self.df.iloc[indices[0]]

# Setting up Environment Variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")  

model = ChatGoogleGenerativeAI(
    model="gemini-pro", 
    google_api_key=google_api_key,
    temperature=0.5
)

# function to get vector store

def create_csv_agent(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return create_pandas_dataframe_agent(
        model, 
        df, 
        verbose=True,
        handle_parsing_errors = True,
    )

# function to get context retriever chain
def get_context_retriever_chain(csv_agent):

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get the informtion relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(model, csv_agent, prompt)
    return retriever_chain

# function to get context retriever chain
def get_conversational_rag_chain(retriever_chain):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# function to handle conversation
def handle_conversation(user_question):
    if "vector_store" not in st.session_state:
        raise ValueError("CSV agent is not initialised")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am ChickIn Chat. How can I help you?")]

    st.session_state.chat_history.append(HumanMessage(content=user_question))


# function to get response 
def get_response(user_question):
    if "vector_store" not in st.session_state:
        raise ValueError("CSV agent is not initialised")
    
    # Ensure chat_history is correctly handled and passed to the conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Properly format the input for the conversation chain
    input_data = {
        "chat_history": st.session_state.chat_history,
        "input": user_question
    }

    response = conversation_rag_chain.invoke(input_data)
    return response['answer']



def append_message(user_question):
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    response = get_response(user_question)
    st.session_state.chat_history.append(AIMessage(content=response))




# Streamlit UI
st.set_page_config(page_title="ChickIn Chat", page_icon="🐥")
st.header("ChickIn Chat")

# sidebar to upload csv
with st.sidebar:
    st.header("Settings")
    user_csv = st.file_uploader("Upload your CSV file", type=["csv"])
    

if user_csv is not None:

    vector_store = create_csv_agent(user_csv)
    st.session_state.vector_store = vector_store

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am ChickIn Chat. How can I help you?")
        ]

    # user query
    user_question = st.chat_input("Input your question here:")
    if user_question is not None and user_question != "":
        append_message(user_question)

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


