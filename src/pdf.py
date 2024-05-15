'''import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core import SimpleDirectoryReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress = True)
        index.storage_context(persist_dir = index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir = index_name)
        )
    return index


pdf_path = "data/Poultry.pdf"
poultry_pdf = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
poultry_index = get_index(poultry_pdf, "poultry_index")
poultry_engine = poultry_index.as_query_engine()
'''