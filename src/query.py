
from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.gemini import Gemini
from src.prompts import pandas_prompt_str, instruction_str, response_synthesis_prompt_str
from llama_index.core.query_pipeline import (
    QueryPipeline,
    Link,
    InputComponent,
)

from src.ResponseWithChatHistory import ResponseWithChatHistory
from llama_index.postprocessor.colbert_rerank import ColbertRerank

## INITIALISING MODEL AND READING DATA
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") 

llm = Gemini(model="models/gemini-pro")
csv_file_path = 'data/bigDataHoliday.csv'
df = pd.read_csv(csv_file_path)

'''## PANDAS AI
from pandasai import SmartDataframe
from langchain_google_genai import GoogleGenerativeAI

langchain_llm = GoogleGenerativeAI(google_api_key=google_api_key,)
pandasData = SmartDataframe("data/bigDataHoliday.csv", config={"llm": langchain_llm})
'''

## PROMPT TEMPLATES
pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)


## LLM TO REWRITE USER QUERY
'''rewrite = (
    "Rewrite the query to a semantic search engine using the current conversation.\n"
    "\n"
    "\n"
    "{chat_history_str}"
    "\n"
    "Latest Message: {query_str}"
    "Query:"
)

rewrite_template = PromptTemplate(rewrite)'''

'''## CHAT HISTORY
response_component = ResponseWithChatHistory(
    llm=llm, 
    system_prompt = (
        " You are a Chat Bot designed to help the farmers in the poultry industry in Indonesia." 
        "You will be provided with the previous chat history, and answer the user's query "
        "as well as possibly relevant context."
    )
)'''



## PIPELINE CONSTRUCTION

qp = QueryPipeline(
    modules={
        "input": InputComponent(),
        "pandas_prompt": pandas_prompt,
        "llm1": llm,
        "pandas_output_parser": pandas_output_parser,
        "response_synthesis_prompt": response_synthesis_prompt,
        "llm2": llm,
    },
    verbose=True,
)

qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
qp.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key="query_str"),
        Link(
            "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
        ),
        Link(
            "pandas_output_parser",
            "response_synthesis_prompt",
            dest_key="pandas_output",
        ),
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")

