import os
import re
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# importing engine libraries
from prompts import instruction_str, new_prompt, context
from note_engine import note_engine
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.experimental.query_engine import PandasQueryEngine


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") 

# initialising model
'''model = ChatGoogleGenerativeAI(
        model="gemini-pro", 
        google_api_key=google_api_key,
        temperature=0.5
    )'''

llm = Gemini(model="models/gemini-pro")


# loading data
csv_file_path = '/workspaces/ChickIn_Chat/data/bigDataHoliday.csv'
data = pd.read_csv(csv_file_path)

# initialising dataframe agent
csv_agent = PandasQueryEngine(data, llm = llm,  verbose = True, instruction_str = instruction_str)
csv_agent.update_prompts({"pandas_prompt": new_prompt})


# creating new list of tools for bot to use
tools = [
    note_engine, 
    QueryEngineTool(
        query_engine = csv_agent, metadata = ToolMetadata(
            name = "avg_price_data", 
            description = "this gives information about the average selling price of the chickens from the different provinces and units from 2019 to 2023 in Indonesia. It has information about the average body weight, supply and demand and whether that day is a holiday or not.  "
        ),
    ),
]


agent = ReActAgent.from_tools(tools, llm = llm, verbose = True, context = context)


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)