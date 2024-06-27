
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompts import refine_template, pre_csv_template
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.prompts import PromptTemplate






## INITIALISING MODEL AND READING DATA
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") 

llm = ChatGoogleGenerativeAI(api_key=google_api_key, model = "models/gemini-1.5-pro", temperature = 0)

# csv data
csv_file_path = "data/Big Data Holiday.csv"
df = pd.read_csv(csv_file_path)


## PROMPT TEMPLATES
'''prompt = ChatPromptTemplate.from_template(prompt_template_1)

response_synthesis_prompt = ChatPromptTemplate.from_template(response_synthesis_prompt_str)'''

pre_csv_prompt = ChatPromptTemplate.from_template(pre_csv_template)
#custom_prefix = "Have a conversation with a human, Answer step by step and use the dataset that has been given to you. s Now, You have access to the following tool:"

csv_agent  = create_pandas_dataframe_agent(llm, 
                                           df = df,
                                           agent_type = "zero-shot-react-description",
                                           verbose = True, 
                                           return_intermediate_steps=False)
chain_one = LLMChain(llm=llm, prompt=pre_csv_prompt)




## CHAIN

refine_prompt = ChatPromptTemplate.from_template(refine_template)
chain_two = LLMChain(llm = llm, prompt = refine_prompt)



main_chain = SimpleSequentialChain(
                             chains=[chain_one, csv_agent, chain_two], verbose=True)


    




