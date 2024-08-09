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
from app import INDEX_MAPPING, PROJECT_PATH
from app.models.model import embed_model, llm
from app.utils.prompt_templates import (
    IMG_PROMPT_TMPL,
    AU_PROMPT_TMPL,
    VID_PROMPT_TMPL,
    DOC_PROMPT_TMPL,
)
from app.utils.node_postprocessors import reorder, reranker, metadatareplacement
from app.services.ingest_data import build_ingestion_pipeline
from deep_translator import GoogleTranslator
import detectlanguage

Settings.embed_model = embed_model
Settings.llm = llm

detectlanguage.configuration.api_key = os.environ.get('DEEPTRANSLATOR_API')


def load_vetor_store_index(index_name):
    pipeline = build_ingestion_pipeline(index_name)
    doctype = INDEX_MAPPING.get(index_name)
    storage_path = os.path.join(os.path.join(PROJECT_PATH, "pipeline_storage", doctype))
    pipeline.load(storage_path)
    vector_store = pipeline.vector_store
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index

def define_query_engine(index_name, prompt_template, node_postprocessors, output_cls, response_mode="simple_summarize", topk=5):
    index = load_vetor_store_index(index_name)
    prompt = PromptTemplate(prompt_template)

    query_engine = index.as_query_engine(
        response_mode=response_mode, use_async=True,
        text_qa_template=prompt, similarity_top_k=topk,
        node_postprocessors=node_postprocessors,
    )
    return query_engine

CHOICES = [
    "This tool answers specific questions about the image files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the audio files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the video files. Consider using this tool when the user does not mention any document type in the query.",
    "This tool answers specific questions about the text files (not image, audio or video). Consider using this tool when the user does not mention any document type in the query.",
]

def construct_router_query_engine(choices=CHOICES):
    img_query_engine = define_query_engine(index_name="img-index",
                                           prompt_template=IMG_PROMPT_TMPL,
                                           node_postprocessors=[reranker])
    au_query_engine = define_query_engine(index_name="au-index",
                                          prompt_template=AU_PROMPT_TMPL,
                                          node_postprocessors=[reranker])
    vid_query_engine = define_query_engine(index_name="vid-index",
                                          prompt_template=VID_PROMPT_TMPL,
                                          node_postprocessors=[reranker])
    doc_query_engine = define_query_engine(index_name="doc-index",
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
  ids = []
  for key, value in response.metadata.items():
    try:
      id = value.get('ID')
      if (str(id) in response.response) and (id not in ids):
        source.append({'id': id, 'title': value.get('TITLE')})
      ids.append(id)
    except:
      continue
  return source

def structured_output(text, response):
  answer = {}
  answer['answer'] = text
  answer['source'] = get_source_in_resonse(response)
  answer['detail'] = get_metadata_in_response(response)
  return answer

def retrieve_files(query_engine, query, data_path):
  ingested_nodes = ingest(data_path)
  lang = detectlanguage.detect(query)[0]['language']
  print("QUERY LANG", lang)
  response = query_engine.query(query)
  response_text = response.response
  if lang=='ko':
    response_text = GoogleTranslator(target='ko').translate(response_text)
  output = structured_output(response_text, response)
  return output
