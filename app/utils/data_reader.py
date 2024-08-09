from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode, IndexNode
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.readers.base import BaseReader
import re
import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
from app import PROJECT_PATH

DATA_PATH = os.path.join(PROJECT_PATH, "data/llm_analyze_data")

def get_value(dictionary, key):
    # Get the value associated with the key, default to '' if key is not found
    value = dictionary.get(key)

    # Return '' if value is None, otherwise return the value
    return '' if value is None else value

def del_key(dictionary, key):
    # Check if the key exists in the dictionary
    dictionary.pop(key, None)
    return dictionary

def classify_documents(documents):
    """
    Classifies a list of documents into categories based on their file type.

    Args:
        documents (list): List of Document objects with metadata.

    Returns:
        dict: Dictionary with categorized documents.
    """
    # Initialize a defaultdict to store categorized documents
    categorized_docs = defaultdict(list)

    # Iterate through the documents and classify them by file type
    for document in documents:
        file_type = document.metadata.get("file_type", "unknown")
        categorized_docs[file_type].append(document)

    # Convert defaultdict to regular dictionary for output
    return dict(categorized_docs)

class MyFileReader(BaseReader):
    def __init__(self):
        super().__init__()

    def load_data(self, file, extra_info=None):
        """Load and process data from a file."""
        with open(file, "r") as f:
            text = f.read()

        json_data = json.loads(text)
        file_type = self._get_file_type(file.name)
        metadata = self._extract_metadata(json_data, file_type)

        if extra_info:
            extra_info.update(metadata)

        text_content = self._get_text_content(json_data, file_type)
        return [Document(text=text_content, metadata=extra_info or {})]

    def _get_file_type(self, file_name):
        """Determine the file type based on the file name."""
        if file_name.startswith("image"):
            return "image"
        elif file_name.startswith("document"):
            return "document"
        elif file_name.startswith("audio"):
            return "audio"
        elif file_name.startswith("video"):
            return "video"
        else:
            raise ValueError("Unsupported file type")

    def _extract_metadata(self, json_data, file_type):
        """Extract metadata from the JSON data."""
        metadata = {
            "file_id": get_value(json_data, "id"),
            "md5": get_value(json_data, "md5"),
            "file_name": get_value(json_data, "originalName"),
            "file_extension": get_value(json_data, "extension"),
            "file_type": file_type,
            "file_size": get_value(json_data, "size"),
            "height": get_value(json_data, "height"),
            "width": get_value(json_data, "width"),
            "duration": get_value(json_data, "duration"),
            "density": get_value(json_data, "density"),
            "channels": get_value(json_data, "channels"),
            "category": str(get_value(json_data, "category")),
            "people": self._process_people(json_data.get("people")),
            "organizations": self._process_organizations(json_data.get("organizations")),
            "narrationStt": self._process_narrationStt(json_data.get("narrationStt"))
        }

        if file_type in ["audio", "video"]:
            metadata["stt_summary"] = json_data.get("stt", {}).get("sttSummary") or ""
        if file_type == "document":
            metadata["description"] = get_value(json_data,"desc" )

        return metadata

    def _get_text_content(self, json_data, file_type):
        """Extract the text content based on the file type."""
        if file_type == "image":
            return json_data.get("desc") or ""
        elif file_type == "document":
            return json_data.get("textData") or ""
        elif file_type in ["audio", "video"]:
            return json_data.get("stt", {}).get("sttData") or ""
        else:
            return ""

    def _process_narrationStt(self, lst):
        """Process the narrationStt field from the JSON data."""
        if not lst:
            return ""
        else:
          narrationStt = {}
          for item in lst:
              text = [entry["content"] for entry in json.loads(item["sttData"])]
              narrationStt[f"user_{ get_value(item, 'id')}"] = text
          return str(narrationStt)

    def _process_people(self, lst):
        """Process the people field from the JSON data."""
        if not lst:
            return ""
        else:
            people = [{"id": get_value(item, "id"), "nickName": get_value(item, "nickName"), "name": get_value(item, "name"), "dateOfBirth": get_value(item, "dateOfBirth")} for item in lst]
            return str(people)

    def _process_organizations(self, lst):
        """Process the organizations field from the JSON data."""
        if not lst:
            return ""
        else:
            org = [{"id": get_value(item, "id"), "name": get_value(item, "name"), "otherName": get_value(item, "otherName")} for item in lst]
        return str(org)

def read_file(path):
    """Read files from the specified directory and process them."""
    reader = SimpleDirectoryReader(input_dir=path, file_extractor={".json": MyFileReader()})
    documents = reader.load_data()
    for doc in documents:
        doc.excluded_embed_metadata_keys = []
        doc.excluded_llm_metadata_keys = []
        doc.id_ = doc.metadata["file_id"]
    return documents

class ImageConstructor:
    def __init__(self, documents):
        self.documents = documents
        self.chunk_size = 512
        self.chunk_overlap = 10
        self.separator = "\n\n"
        self.document_map = {}

    def construct_nodes(self):
        for doc_idx, page in enumerate(self.documents):
            text = page.text
            metadata = page.metadata

            # Add document to document_map
            doc_id = str(metadata.get("file_id"))
            self.document_map[doc_id] = page

        """Constructs nodes from the documents."""
        text_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator)
        img_nodes = text_parser.get_nodes_from_documents(self.documents)
        for node in img_nodes:
            node.id_ = node.metadata["file_id"]

            # Add SOURCE relationship
            document = self.document_map.get(node.id_)
            if document:
              node_source = document.as_related_node_info()
              node_source.metadata = {}
              node.relationships[NodeRelationship.SOURCE] = node_source
                
        return img_nodes
    
class AudioVideoNodeConstructor:
    def __init__(self, documents):
        self.documents = documents
        self.document_map = {}
        self.parrent_chunk = {}
        self.child_chunk = {}
        self.nodes_map = {}
        self.parrent_nodes = {}

    def process_stt(self, json_string):
        # Parse the JSON string
        data = json.loads(json_string)

        # Initialize the list to store the formatted data
        formatted_data = []

        # Iterate through each item in the data
        for item in data:
            # Split the "time" field into "start_time" and "end_time"
            times = item["time"].split(" --> ")
            start_time = times[0]
            end_time = times[1]

            # Extract the "content"
            content = get_value(item, "content")

            # Extract the "speaker"
            speaker = get_value(item, "speaker")

            # Create a dictionary with the formatted data
            formatted_item = {
                "start_time": start_time,
                "end_time": end_time,
                "content": content,
                "speaker": speaker
            }

            # Append the formatted item to the list
            formatted_data.append(formatted_item)

        return formatted_data

    def process_and_construct_chunks(self):
        for doc_idx, page in enumerate(self.documents):
            text = page.text
            metadata = page.metadata

            # Add document to document_map
            doc_id = metadata.get("file_id")
            self.document_map[doc_id] = page

            desc = get_value(metadata, "description")
            del_key(metadata, "description")

            self.parrent_chunk[doc_idx] = {
                "text": desc,
                "metadata": metadata
            }

            del_key(metadata, "stt_summary")
            stt = self.process_stt(text)

            if len(stt) == 0:
                self.child_chunk[doc_idx] = {}
                continue
            else:
                self.child_chunk[doc_idx] = []
                for i in range(len(stt)):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["start_time"] = stt[i].get("start_time")
                    chunk_metadata["end_time"] = stt[i].get("end_time")
                    chunk_metadata["speaker"] = stt[i].get("speaker")
                    cur_text_chunks = stt[i].get("content")

                    self.child_chunk[doc_idx].append({
                        "text": cur_text_chunks,
                        "metadata": chunk_metadata
                    })

    def create_nodes_from_chunks(self):
        for idx in self.parrent_chunk.keys():
            parrent_node = TextNode(
                text=self.parrent_chunk[idx]["text"],
                metadata=self.parrent_chunk[idx]["metadata"],
                id_=str(self.parrent_chunk[idx]["metadata"]["file_id"])
            )
            self.parrent_nodes[parrent_node.id_] = parrent_node

            if idx not in self.child_chunk or len(self.child_chunk[idx]) == 0:
                self.nodes_map[parrent_node.id_] = []
            else:
                child_nodes = []
                for i, chunk in enumerate(self.child_chunk[idx]):
                    child_node = TextNode(
                        text=chunk["text"],
                        metadata=chunk["metadata"],
                        id_=str(chunk["metadata"]["file_id"]) + '_' + str(i + 1)
                    )
                    child_nodes.append(child_node)
                self.nodes_map[parrent_node.id_] = child_nodes

    def index_nodes(self):
        nodes = []

        for node_id in self.nodes_map.keys():
            parrent = self.parrent_nodes[node_id]
            child = self.nodes_map[node_id]
            child_lst = []

            # Add SOURCE relationship
            document = self.document_map.get(parrent.metadata["file_id"])
            if document:
                parrent_source = document.as_related_node_info()
                parrent_source.metadata = {}
                parrent.relationships[NodeRelationship.SOURCE] = parrent_source

            for i in range(len(child)):
                child_source = parrent.as_related_node_info()
                child_source.metadata = {}
                child[i].relationships[NodeRelationship.PARENT] = child_source

                if i == 0:
                    if len(child) > 1:
                        next_source = child[i + 1].as_related_node_info()
                        next_source.metadata = {}
                        child[i].relationships[NodeRelationship.NEXT] = next_source
                elif i == len(child) - 1:
                    prev_source = child[i - 1].as_related_node_info()
                    prev_source.metadata = {}
                    child[i].relationships[NodeRelationship.PREVIOUS] = prev_source
                else:
                    next_source = child[i + 1].as_related_node_info()
                    next_source.metadata = {}
                    prev_source = child[i - 1].as_related_node_info()
                    prev_source.metadata = {}
                    child[i].relationships[NodeRelationship.NEXT] = next_source
                    child[i].relationships[NodeRelationship.PREVIOUS] = prev_source
                
                child_info = child[i].as_related_node_info()
                child_info.metadata = {}
                child_lst.append(child_info)

            parrent.relationships[NodeRelationship.CHILD] = child_lst
            nodes.append(parrent)
            nodes.extend(child)

        return nodes

    def construct_nodes(self):
        self.process_and_construct_chunks()
        self.create_nodes_from_chunks()
        return self.index_nodes()

    
class DocumentConstructor:
    def __init__(self, documents):
        self.documents = documents
        self.chunk_size = 1024
        self.chunk_overlap = 10
        self.separator = "\n\n"
        self.document_map = {}
        self.nodes_map = {}
        self.parrent_nodes = {}

    def process_and_construct_chunks(self):
        """Processes documents into chunks and constructs nodes from them."""
        text_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator)
        doc_nodes = text_parser.get_nodes_from_documents(self.documents)

        for node in doc_nodes:
            node.id_ = node.metadata["file_id"]
            node.relationships = {}
            if node.id_ in self.nodes_map:
                self.nodes_map[node.id_].append(node)
            else:
                self.nodes_map[node.id_] = [node]

    def create_parrent_node(self):
        """Creates parent nodes from the documents."""
        for page in self.documents:
            self.document_map[page.id_] = page
            metadata = page.metadata
            desc = get_value(metadata, "description")
            del_key(metadata, "description")

            parrent_node = TextNode(
                text=desc,
                metadata=metadata,
                id_=str(metadata["file_id"])
            )
            self.parrent_nodes[parrent_node.id_] = parrent_node

    def index_nodes(self):
        """Indexes nodes and sets relationships between them."""
        nodes = []

        for node_id in self.nodes_map.keys():
            parrent = self.parrent_nodes[node_id]
            children = self.nodes_map[node_id]

            child_lst = []

            # Add SOURCE relationship
            document = self.document_map.get(parrent.id_)
            if document:
                parrent_source = document.as_related_node_info()
                parrent_source.metadata = {}
                parrent.relationships[NodeRelationship.SOURCE] = parrent_source
            for i, child in enumerate(children):
                child.id_ = f"{child.metadata['file_id']}_{i + 1}"

                child_source = parrent.as_related_node_info()
                child_source.metadata = {}
                child.relationships[NodeRelationship.PARENT] = child_source

                if i == 0:
                    if len(children) > 1:
                        next_source = children[i + 1].as_related_node_info()
                        next_source.metadata = {}
                        child.relationships[NodeRelationship.NEXT] = next_source
                elif i == len(children) - 1:
                    prev_source = children[i - 1].as_related_node_info()
                    prev_source.metadata = {}
                    child.relationships[NodeRelationship.PREVIOUS] = prev_source
                else:
                    next_source = children[i + 1].as_related_node_info()
                    next_source.metadata = {}
                    prev_source = children[i - 1].as_related_node_info()
                    prev_source.metadata = {}
                    child.relationships[NodeRelationship.NEXT] = next_source
                    child.relationships[NodeRelationship.PREVIOUS] = prev_source

                child_info = child.as_related_node_info()
                child_info.metadata = {}
                child_lst.append(child_info)


            parrent.relationships[NodeRelationship.CHILD] = child_lst
            nodes.append(parrent)
            nodes.extend(children)

        return nodes

    def construct_nodes(self):
        """Constructs the full set of nodes from the documents."""
        self.process_and_construct_chunks()
        self.create_parrent_node()
        return self.index_nodes()