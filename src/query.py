
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
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from src.prompts import context
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.agent import ReActAgent

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.embeddings.gemini import GeminiEmbedding


## INITIALISING MODEL AND READING DATA
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") 
llm = Gemini(api_key=google_api_key, model="models/gemini-pro")

# csv data
csv_file_path = "data/Big Data Holiday.csv"
df = pd.read_csv(csv_file_path)


## PROMPT TEMPLATES
pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)


## TOOLS

# pandas query engine
csv_engine = PandasInstructionParser(df)

# agent tool
tools = [
    QueryEngineTool(
        query_engine = csv_engine, 
        metadata = ToolMetadata(
            name = "bigData",
            description = " this gives information of the sales price and average body weight of the chickens across different areas and units from 2019 to 2023."
        )
    ), 
    
        
]

agent = ReActAgent.from_tools(tools, llm = llm, verbose = True, context = context)



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
        )
    ]
)
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")

