import weaviate
from typing import Dict, Any
import os
# import openai

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from deep_translator import GoogleTranslator
import detectlanguage
# from llama_index.llms.openai import OpenAI

from app.models.model import embed_model, llm
from app.utils.node_postprocessors import *
from app.utils.prompt_templates import QUERY_REWRITE_PROMPT_TMPL, CONTENT_GEN_PROMPT_TMPL

Settings.embed_model = embed_model
Settings.llm = llm

# os.environ["OPENAI_API_KEY"] = "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"
# openai.api_key = os.environ["OPENAI_API_KEY"]
# LLM = OpenAI(model="gpt-3.5-turbo")
detectlanguage.configuration.api_key = "1fc477ea3d0e0c984834610b3f245d30"

# Vector DB
# cloud
auth_config = weaviate.AuthApiKey(
    api_key=os.environ.get('WEAVIATE_API_KEY')
)
client = weaviate.Client(
    os.environ.get('WEAVIATE_URL'),
    auth_client_secret=auth_config,
)

def load_vetor_store_index(index_name):
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=index_name)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


# class RewriteQuery(BaseQueryTransform):
#     def __init__(
#         self,
#         llm: Optional[LLMPredictorType] = None,
#         rewrite_query_prompt: Optional[BasePromptTemplate] = None,
#         verbose: bool = False,
#     ) -> None:
#         """Init params."""
#         super().__init__()
#         self._llm = llm
#         self._rewrite_query_prompt = rewrite_query_prompt

#     def _get_prompts(self) -> PromptDictType:
#         """Get prompts."""
#         return {"rewrite_query_prompt": self._rewrite_query_prompt}

#     def _update_prompts(self, prompts: PromptDictType) -> None:
#         """Update prompts."""
#         if "rewrite_query_prompt" in prompts:
#             self._rewrite_query_prompt = prompts["rewrite_query_prompt"]

#     def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
#         # given the text from the index, we can use the query bundle to generate
#         # a new query bundle
#         query_str = query_bundle.query_str
#         new_query_str = self._llm.predict(
#             self._rewrite_query_prompt,
#             query_str=query_str,
#         )

#         print(f"> Current query: {query_str}\n")
#         print(f"> New query: {new_query_str}\n")

#         return QueryBundle(
#             query_str=new_query_str,
#             custom_embedding_strs=[new_query_str],
#         )


def get_id_in_resonse(response):
  ids = []
  for key, value in response.metadata.items():
    try:
      id = value.get('ID')
      ids.append(id)
    except:
      continue
  return list(set(ids))
 

def generate_content(query_engine, query):
    lang = detectlanguage.detect(query)[0]['language']
    print("QUERY LANG", lang)
    response = query_engine.query(query)
    response_text = response.response
    answer = {}
    if lang=='ko':
        response_text = GoogleTranslator(target='ko').translate(response_text)
    answer['answer'] = response_text
    answer['ids'] = get_id_in_resonse(response)
    answer['sources'] = response.metadata
    return answer


# query_rewrite_prompt = PromptTemplate(QUERY_REWRITE_PROMPT_TMPL)
content_gen_prompt = PromptTemplate(CONTENT_GEN_PROMPT_TMPL, prompt_type=PromptType.SUMMARY)

index = load_vetor_store_index("LlamaIndex_doc_index")
# query_transform = RewriteQuery(llm=LLM, rewrite_query_prompt=query_rewrite_prompt)

query_engine_generate = index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
    llm=llm,
    similarity_top_k=5,
    summary_template=content_gen_prompt,
    node_postprocessors = [reranker, metadatareplacement, hide_metadata, reorder]
)
# transform_query_engine = TransformQueryEngine(query_engine_generate, query_transform)