from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    LongContextReorder,
    MetadataReplacementPostProcessor
)
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List, Optional



reorder = LongContextReorder()
reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
metadatareplacement = MetadataReplacementPostProcessor(target_metadata_key="window")

class HideMetadataPostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        for node_with_score in nodes:
            node = node_with_score.node
            metadata = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = metadata

        return nodes

hide_metadata = HideMetadataPostprocessor()