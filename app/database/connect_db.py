import weaviate
from pinecone import Pinecone, ServerlessSpec, PodSpec
import os
import logging

def create_index_pinecone(pc,
                          index_name,
                          index_type="serverless",
                          dimension=1024,
                          metric="cosine",
                          cloud="aws",
                          region="us-east-1",
                          environment='us-east1-gcp',
                          pod_type='p1.x1',
                          pods=1,
                          api_key=os.environ.get("PINECONE_API_KEY")):
    if index_type == "serverless":
        spec=ServerlessSpec(cloud=cloud, region=region)
    elif index_type == "podbased":
        spec=PodSpec(
        		environment=environment,
        		pod_type=pod_type,
        		pods=pods
        )
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec,
        )
    except Exception as e:
        logging.exception(f"Error creating index with name {index_name}")

def connect_pinecone(api_key=os.environ.get("PINECONE_API_KEY")):
    pc = Pinecone(api_key=api_key)
    return pc

def connect_weaviate(api_key=os.environ.get("WEAVIATE_API_KEY"), cluster_url=os.environ.get("WEAVIATE_CLUSTER_URL")):
    client = weaviate.connect_to_wcs(
        cluster_url=cluster_url,
        auth_credentials=weaviate.auth.AuthApiKey(api_key))
    return client

def connect_elasticsearch(cloud_id=os.environ.get("ELASTICSEARCH_CLOUD_ID"), api_key=os.environ.get("ELASTICSEARCH_API_KEY")):
    return [cloud_id, api_key]

def connect_to_database(db_name):
    if db_name == "pinecone":
        client = connect_pinecone()
    elif db_name == "weaviate":
        client = connect_weaviate()
    elif db_name == "elasticsearch":
        client = connect_elasticsearch()
    return client