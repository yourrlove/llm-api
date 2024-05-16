from . import api_blueprint
from flask import Flask, jsonify, request, render_template
from flask_restful import Resource, Api
import logging
import os
from llama_index.core import Settings
from app.database.index import define_vector_store_index
from app.utils.data_reader import read_image, read_audio, read_video, read_document
from app.services.retrieve_assets import construct_router_query_engine, retrieve_files
from app.services.generate_content import generate_content, query_engine_generate
from app import PROJECT_PATH
from app.models.model import llm, embed_model

@api_blueprint.route('/ai/v1/llm/assets/<query>', methods=['GET'])
def get_response_by_query(query):
 query_engine = construct_router_query_engine()
 output = retrieve_files(query_engine, query)
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
            file_path = request.args.get('file_path')
            db = request.args.get('db_name')
            file_path = os.path.join(PROJECT_PATH, file_path)
            if db == 'LlamaIndex_img_index':
              try:
                img_nodes = read_image(file_path)
                define_vector_store_index("LlamaIndex_img_index", img_nodes, llm, embed_model)
                print(jsonify({"success": True}))
                return jsonify({"success": True})
              except Exception as e:
                logging.exception(e)
                return jsonify({"success": False})
            if db == 'LlamaIndex_au_index':
              try:
                au_nodes = read_audio(file_path)
                define_vector_store_index("LlamaIndex_au_index", au_nodes, llm, embed_model)
                print(jsonify({"success": True}))
                return jsonify({"success": True})
              except Exception as e:
                logging.exception(e)
                return jsonify({"success": False})
            if db == 'LlamaIndex_vid_index':
              try:
                vid_nodes = read_video(file_path)
                define_vector_store_index("LlamaIndex_vid_index", vid_nodes, llm, embed_model)
                print(jsonify({"success": True}))
                return jsonify({"success": True})
              except Exception as e:
                logging.exception(e)
                return jsonify({"success": False})
            if db == 'LlamaIndex_doc_index':
              try:
                doc_nodes = read_document(file_path)
                define_vector_store_index("LlamaIndex_doc_index", doc_nodes, llm, embed_model)
                print(jsonify({"success": True}))
                return jsonify({"success": True})
              except Exception as e:
                logging.exception(e)
                return jsonify({"success": False})
        

