from app.utils.data_reader import *
from app.utils.ingestion_pipeline import CustomIngestionPipeline
from app.database.connect_db import connect_to_database, create_index_pinecone
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.pinecone import PineconeVectorStore

import logging
import os
from collections import defaultdict
from app.models.model import embed_model as EMBED_MODEL
from app import INDEX_MAPPING

def load_data(file_path):
    # Read and classify documents
    documents = read_file(file_path)
    classified_docs = classify_documents(documents)

    constructor_mapping = {
        'image': ImageConstructor,
        'audio': AudioVideoNodeConstructor,
        'video': AudioVideoNodeConstructor,
        'document': DocumentConstructor,
    }

    classified_nodes = defaultdict(list)

    # Process each document type
    for doc_type, Constructor in constructor_mapping.items():
        if classified_docs.get(doc_type):
            try:
                constructor = Constructor(classified_docs.get(doc_type))
                nodes = constructor.construct_nodes()
                classified_nodes[doc_type] = nodes
            except Exception as e:
                logging.exception(f"Error processing documents for {doc_type}")

    return classified_nodes

def build_ingestion_pipeline(index_name, db_name="pinecone"):
    db_client = connect_to_database(db_name)
    if db_name == "pinecone":
        if index_name not in db_client.list_indexes().names():
            create_index_pinecone(pc=db_client, index_name=index_name)
        vector_store = PineconeVectorStore(pinecone_index=db_client.Index(index_name))
    elif db_name == "weaviate":
        vector_store = WeaviateVectorStore(weaviate_client=db_client, index_name=index_name)
    elif db_name == "elasticsearch":
        vector_store = ElasticsearchStore(
            index_name=index_name,
            es_cloud_id=db_client[0],
            es_api_key=db_client[1]
        )

    pipeline = CustomIngestionPipeline(
        transformations=[EMBED_MODEL],
        docstore=SimpleDocumentStore(),
        vector_store=vector_store,
        cache=IngestionCache(),
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    return pipeline

def ingest(file_path):

    try:
        classified_nodes = load_data(file_path=file_path)
    except Exception as e:
        logging.exception(f"Error reading and classifying documents")
        return

    classified_ingested_nodes = {}

    for index_name, doc_type in INDEX_MAPPING.items():
        nodes = classified_nodes.get(doc_type)
        storage_path = os.path.join(PROJECT_PATH, "pipeline_storage", doc_type)
        try:
            pipeline = build_ingestion_pipeline(index_name)
        except Exception as e:
            logging.exception(f"Error building ingestion pipeline for {doc_type}")
            return

        if os.path.exists(storage_path):
            logging.info(f"Restore {doc_type} ingestion pipeline from {storage_path}")
            pipeline.load(storage_path)
        try:
            ingested_nodes = pipeline.run(nodes=nodes, show_progress=True)
            logging.info(f"Ingested {len(ingested_nodes)} {doc_type} Nodes")
            classified_ingested_nodes[doc_type] = len(ingested_nodes)
            pipeline.persist(storage_path)
        except Exception as e:
            logging.exception(f"Error running ingestion pipeline for {doc_type}")
    
    return classified_ingested_nodes