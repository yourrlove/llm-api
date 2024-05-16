from llama_index.core.indices import KeywordTableIndex
import weaviate
import os
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    PromptTemplate
)
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    LongContextReorder,
    MetadataReplacementPostProcessor
)
from app.models.model import embed_model, llm
from app.utils.prompt_templates import (
    IMG_PROMPT_TMPL,
    AU_PROMPT_TMPL,
    VID_PROMPT_TMPL,
    DOC_PROMPT_TMPL,
)
from app.utils.node_postprocessors import reorder, reranker, metadatareplacement
from deep_translator import GoogleTranslator
import detectlanguage

Settings.embed_model = embed_model
Settings.llm = llm

detectlanguage.configuration.api_key = os.environ.get('DEEPTRANSLATOR_API')

# Weaviate VectorDB (cloud)
auth_config = weaviate.AuthApiKey(
    api_key=os.environ.get('WEAVIATE_API_KEY')
)
client = weaviate.Client(
    os.environ.get('WEAVIATE_URL'),
    auth_client_secret=auth_config,
)

# local
# client = weaviate.Client("http://localhost:8080")

def load_vetor_store_index(index_name):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def define_query_engine(index_name, prompt_template, node_postprocessors, response_mode="simple_summarize", topk=5):
    index = load_vetor_store_index(index_name)
    prompt = PromptTemplate(prompt_template)

    query_engine = index.as_query_engine(
        response_mode=response_mode, use_async=True,
        text_qa_template=prompt, similarity_top_k=topk,
        node_postprocessors=node_postprocessors
    )
    return query_engine

CHOICES = [
    "This tool answers specific questions about the image files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the audio files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the video files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the text files (not image, audio or video). Consider using this tool when the user does not mention any document type in the query.",
]

def construct_router_query_engine(choices=CHOICES):
    img_query_engine = define_query_engine(index_name="LlamaIndex_img_index",
                                           prompt_template=IMG_PROMPT_TMPL,
                                           node_postprocessors=[reranker])
    au_query_engine = define_query_engine(index_name="LlamaIndex_au_index",
                                          prompt_template=AU_PROMPT_TMPL,
                                          node_postprocessors=[reranker])
    vid_query_engine = define_query_engine(index_name="LlamaIndex_vid_index",
                                          prompt_template=VID_PROMPT_TMPL,
                                          node_postprocessors=[reranker])
    doc_query_engine = define_query_engine(index_name="LlamaIndex_doc_index",
                                          prompt_template=DOC_PROMPT_TMPL,
                                          node_postprocessors=[reranker, metadatareplacement, reorder])
    
    img_tool = QueryEngineTool.from_defaults(query_engine=img_query_engine, description=choices[0])
    au_tool = QueryEngineTool.from_defaults(query_engine=au_query_engine, description=choices[1])
    vid_tool = QueryEngineTool.from_defaults(query_engine=vid_query_engine, description=choices[2])
    doc_tool = QueryEngineTool.from_defaults(query_engine=doc_query_engine, description=choices[3])

    query_engine = RouterQueryEngine(selector=LLMMultiSelector.from_defaults(),
                                     query_engine_tools=[img_tool, au_tool, vid_tool, doc_tool],
                                    )
    return query_engine

def get_metadata_in_response(response):
  metadata = {}
  for key, value in response.metadata.items():
    try:
      id = value.get('ID')
      if str(id) in response.response:
        metadata[key] = response.metadata[key]
    except:
      continue
  return metadata

def get_source_in_resonse(response):
  source = []
  for key, value in response.metadata.items():
    try:
      id = value.get('ID')
      if str(id) in response.response:
        source.append({'id': id, 'title': value.get('TITLE')})
    except:
      continue
  return source

def structured_output(text, response):
  answer = {}
  answer['answer'] = text
  answer['source'] = get_source_in_resonse(response)
  answer['detail'] = get_metadata_in_response(response)
  return answer

def retrieve_files(query_engine, query):
  lang = detectlanguage.detect(query)[0]['language']
  print("QUERY LANG", lang)
  response = query_engine.query(query)
  response_text = response.response
  if lang=='ko':
    response_text = GoogleTranslator(target='ko').translate(response_text)
  output = structured_output(response_text, response)
  return output