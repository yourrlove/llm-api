from app.models.model import *
import weaviate
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import os

print(os.environ.get('WEAVIATE_API_KEY'))
print(os.environ.get('WEAVIATE_URL'))
auth_config = weaviate.AuthApiKey(
    api_key=os.environ.get('WEAVIATE_API_KEY')
)
client = weaviate.Client(
    os.environ.get('WEAVIATE_URL'),
    auth_client_secret=auth_config,
)


PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
#PROJECT_PATH = "/content/drive/MyDrive/CustomLLM/custom_llm"

IMG_PATH = os.path.join(PROJECT_PATH, "data/Images")
VID_PATH = os.path.join(PROJECT_PATH, "data/Videos")
AUDIO_PATH = os.path.join(PROJECT_PATH, "data/Audios")
NEWS_PATH = os.path.join(PROJECT_PATH, "data/News")
PDF_PATH = os.path.join(PROJECT_PATH, "data/pdf")



def define_vector_store_index(index_name, nodes, llm, embed_model):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model
    )