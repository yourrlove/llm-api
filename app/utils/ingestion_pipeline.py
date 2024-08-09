from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from typing import List

class CustomIngestionPipeline(IngestionPipeline):
    def _handle_upserts(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in nodes:
            node_id = node.id_
            ids_from_nodes.add(node_id)
            existing_hash = self.docstore.get_document_hash(node_id)
            if not existing_hash:
                # node doesn't exist, so add it
                deduped_nodes_to_run[node_id] = node
            elif existing_hash and existing_hash != node.hash:
                self.docstore.delete_ref_doc(node_id, raise_error=False)

                if self.vector_store is not None:
                    self.vector_store.delete(node_id)

                deduped_nodes_to_run[node_id] = node
            else:
                continue  # node exists and is unchanged, so skip it

        if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            existing_node_ids_before = set(
                self.docstore.get_all_document_hashes().values()
            )
            node_ids_to_delete = existing_node_ids_before - ids_from_nodes
            for node_id in node_ids_to_delete:
                self.docstore.delete_document(node_id)

                if self.vector_store is not None:
                    self.vector_store.delete(node_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        self.docstore.set_document_hashes({n.id_: n.hash for n in nodes_to_run})
        self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run


    async def _ahandle_upserts(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in nodes:
            node_id = node.id_
            ids_from_nodes.add(node_id)
            existing_hash = await self.docstore.aget_document_hash(node_id)
            if not existing_hash:
                # node doesn't exist, so add it
                deduped_nodes_to_run[node_id] = node
            elif existing_hash and existing_hash != node.hash:
                await self.docstore.adelete_ref_doc(node_id, raise_error=False)

                if self.vector_store is not None:
                    await self.vector_store.adelete(node_id)

                deduped_nodes_to_run[node_id] = node
            else:
                continue  # node exists and is unchanged, so skip it

        if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            existing_node_ids_before = set(
                (await self.docstore.aget_all_document_hashes()).values()
            )
            node_ids_to_delete = existing_node_ids_before - ids_from_nodes
            for node_id in node_ids_to_delete:
                await self.docstore.adelete_document(node_id)

                if self.vector_store is not None:
                    await self.vector_store.adelete(node_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        await self.docstore.async_add_documents(nodes_to_run, store_text=store_doc_text)
        await self.docstore.aset_document_hashes({n.id_: n.hash for n in nodes_to_run})

        return nodes_to_run