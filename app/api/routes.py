from . import api_blueprint
from flask import Flask, jsonify, request, render_template
from flask_restful import Resource, Api
import logging
import os
from llama_index.core import Settings
from app.utils.data_reader import *
from app.services.retrieve_assets import construct_router_query_engine, retrieve_files
#from app.services.generate_content import generate_content, query_engine_generate
from app.services.ingest_data import ingest
from app import PROJECT_PATH, DATA_DIR, DEFAULT_DATA_PATH
from app.models.model import llm, embed_model

@api_blueprint.route('/ai/v1/llm/assets/<query>', methods=['GET'])
def get_response_by_query(query, data_path=DEFAULT_DATA_PATH):
    query_engine = construct_router_query_engine()
    output = retrieve_files(query_engine, query, data_path=data_path)
    if output is None:
        return jsonify({ 'error': 'FACE_DESCRIPTOR_FAIL'}), 404
    else:
        print(output)
        return jsonify(output)

@api_blueprint.route('/ai/v1/llm/articles/<query>', methods=['GET'])
def generate_content_by_query(query):
    response = generate_content(query_engine_generate, query)
    if response is None:
        return jsonify({ 'error': 'FACE_DESCRIPTOR_FAIL'}), 404
    else:
        print(response)
        return jsonify(response)


@api_blueprint.route('/ai/v1/llm/data')
def upload():
    # Extract parameters from the request
    data_folder_name = request.args.get('data_folder_name')

    if not data_folder_name:
        return jsonify({"success": False, "error": "Missing data_folder_name parameter"})

    # Construct the full file path
    data_path = os.path.join(DATA_DIR, data_folder_name)

    # Read and classify documents
    try:
        ingested_nodes = ingest(data_path)
        return jsonify({"success": True, "ingested_nodes": ingested_nodes})
    except Exception as e:
        logging.exception("Error ingest data")
        return jsonify({"success": False, "error": str(e)})