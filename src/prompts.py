pandas_prompt_str = (
    # explain the dataset
    "The dataset contains information about the chicken farms in Indonesia from December 2019 to 2023.\n"
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Take note of these columns: `Province`, `Unit`, `Year`, `Month`. \n"
    "For the `Province` column, the values are the names of the provinces or regions in Indonesia.\n"
    "For the `Unit` column, the values are the districts that are found in the regions.\n"
    "In the `Year` column, the values represent years ranging from 2019 to 2023. However, only the last two digits of each year are shown. For example, a value of 19 in the `Year` column corresponds to the year 2019."
    "In the `Month` column, the values represent the months of the year. For example, a value of 1 in the `Month` column corresponds to the month January\n\n"


    # explain the query
    "Based on the user's query, you are to determine if the query relates to the dataset.\n"
    "If the query is not related to the dataset, skip the instructions below and provide a response that answers back to the query. \n"

    "If the query is related to the dataset, provide the Python code that will execute the query.\n"


    "Based on the user's query, you are to determine if the query relates to the dataset.\n"

    "If the query is not related to the dataset, skip the instructions below and provide a response. \n"

    "If the query is related to the dataset, provide the Python code that will execute the query."
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"

    "Expression:"
)



instruction_str = (
    "You have been given the following query: {query_str}.\n"
    "Based on the {query_str}, determine if whether it is relating to the dataset."

    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"

)




response_synthesis_prompt_str = (
    "You are a helpful assistant for the chicken farmers in Indonesia."
    "You are to assist them by answering their questions related to the chicken farming industry."

    "From {pandas_instructions}, if the query is not related to the dataset, provide a response that answers the user's query\n\n"

    "Else, given the user's question, refine the response and phrase it in such a simple way that farmers would be able to understand. \n"
    "For example, if the user asks for the average weight of chickens in a specific province, you could respond with: 'The average weight of chickens in `province` is `average_weight` kg.'\n\n"
    "For example, if the user asks for the selling price of chickens in each month, list the selling price for each month in a clean, clear and simple format. Round the value to 2 decimal points\n\n"

    "Please keep in mind that the farmers might not have high level education and might not understand complex terms. \n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"

    "Response: "
)

context = """Purpose: The primary role of this agent is to assist users by providing accurate 
            information about the poultry industry in Indonesia"""


