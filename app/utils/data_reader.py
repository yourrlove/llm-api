from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode, IndexNode
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.readers.base import BaseReader
from pypdf import PdfReader
import pysrt
import re
import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import List
from app import PROJECT_PATH

IMG_LABEL_PATH = os.path.join(PROJECT_PATH, "data/Images/label")
VID_LABEL_PATH = os.path.join(PROJECT_PATH, "data/Videos/label")
AUDIO_LABEL_PATH = os.path.join(PROJECT_PATH, "data/Audios/label")


def sentence_window_node_parser(text_node, window_size=3):
    swindow_node_parser = SentenceWindowNodeParser.from_defaults(window_size=window_size)
    window_nodes = []
    for base_node in text_node:
        sub_nodes = swindow_node_parser.get_nodes_from_documents([base_node])
        window_nodes.extend(sub_nodes)
    return window_nodes

class MyImageReader(BaseReader):
    def load_data(self, file, extra_info=None):
        file_label_name = file.name.split(".")[0] + ".txt"
        file_label_path = IMG_LABEL_PATH + "/" + file_label_name
        with open(file, "r") as f:
            with open(file_label_path, "r") as a:
                text = a.read()
        json_data = json.loads(text).get("description")
        metadata = {"ID": str(uuid.uuid4()), "TITLE": file.name.split('.')[0],"categories": json_data.get("tags")}
        text = json_data.get("caption")
        if extra_info:
            metadata = {**metadata, **extra_info}
            # load_data returns a list of Document objects
        return [Document(text=text, metadata = metadata or {})]

def read_image(img_path, chunk_size=512, chunk_overlap=10, separator="\n\n"):
    text_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    reader = SimpleDirectoryReader(input_dir=img_path, file_extractor={".png": MyImageReader(), ".jpg": MyImageReader(), ".jpeg": MyImageReader()})
    documents = reader.load_data()
    img_nodes = text_parser.get_nodes_from_documents(documents)
    for node in img_nodes:
        node.excluded_llm_metadata_keys = []
    return img_nodes


class MyAudioReader(BaseReader):
    def __init__(
        self,
        *args,
        concat_rows: bool = True,
        col_joiner: str = "; ",
        row_joiner: str = "\n\n",
        pandas_config: dict = {},
        **kwargs
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config
    def load_data(self, file, extra_info=None):
        file_label_name = file.name.split(".")[0] + ".srt"
        file_label_path = AUDIO_LABEL_PATH + "/" + file_label_name

        text_list = []
        with open(file, "rb") as f:
            text=""
            subs = pysrt.open(file_label_path)
            df = pd.DataFrame([{"start": sub.start, "end": sub.end, "text": sub.text} for sub in subs])
        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()
        metadata = {"ID": str(uuid.uuid4()), "TITLE": file.name.split('.')[0]}

        if extra_info:
          metadata = {**metadata, **extra_info}

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=metadata or {}
                )
            ]
        else:
            return [
                Document(text=text, metadata=metadata or {}) for text in text_list
            ]

def extract_text_between_times(text, start, end):
    global flag
    flag = False
    text_list =[]
    time_text_list=text.split("\n\n")
    for item in time_text_list:
        start_time = item.split('; ')[0]
        end_time = item.split('; ')[1]
        if start_time == start:
            flag = True
        if(flag):
            text_list.append(item.split('; ')[2])
            if end_time == end:
                flag = False
    return [' '.join(text_list)]

def manually_construct_nodes_from_audio(documents):
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    metadata_chunk = []
    for doc_idx, page in enumerate(documents):
        file_name = page.metadata.get("file_name").split(".")[0] + ".txt"
        file_label_path = AUDIO_LABEL_PATH + "/" + file_name

        with open(file_label_path, "r") as a:
            text = a.read()
        label = json.loads(text)
        for i in label.keys():
            metadata_chunk.append(label.get(i))
            start_time = label.get(i).get("start_time")
            end_time = label.get(i).get("end_time")

            cur_text_chunks = extract_text_between_times(documents[doc_idx].text, start_time, end_time)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        metadata = metadata_chunk[idx]
        src_doc_idx = doc_idxs[idx]
        src_page = documents[src_doc_idx]
        node = TextNode(
            text=text_chunk,
            metadata=metadata,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=src_page.id_)},
        )

        start_char_idx = src_page.text.find(node.get_content(metadata_mode=MetadataMode.NONE))
        node.metadata.update(src_page.metadata)
        if idx > 0:
            node.relationships[NodeRelationship.PREVIOUS] = nodes[idx - 1].as_related_node_info()
        if idx < len(nodes) - 1:
            node.relationships[NodeRelationship.NEXT] = nodes[idx + 1].as_related_node_info()

        nodes.append(node)
    return nodes

def read_audio(audio_path):
    reader = SimpleDirectoryReader(input_dir=audio_path, file_extractor={".mp3": MyAudioReader()})
    documents = reader.load_data()
    for node in documents:
        node.excluded_llm_metadata_keys = []
    au_nodes = manually_construct_nodes_from_audio(documents)
    au_nodes = sentence_window_node_parser(au_nodes)
    return au_nodes


class MyVideoReader(BaseReader):
    def __init__(
        self,
        *args,
        concat_rows: bool = True,
        col_joiner: str = "; ",
        row_joiner: str = "\n\n",
        pandas_config: dict = {},
        **kwargs
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config
    def load_data(self, file, extra_info=None):
        """Load data from file."""
        file_label_name = file.name.split(".")[0]
        #Load audio content
        file_label_path = VID_LABEL_PATH + "/" + file_label_name + ".srt"
        text_list = []
        with open(file, "rb") as f:
          text=""
          subs = pysrt.open(file_label_path)
          df = pd.DataFrame([{"start": sub.start, "end": sub.end, "text": sub.text} for sub in subs])


        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()

        metadata = {"ID": str(uuid.uuid4()), "TITLE": file.name.split('.')[0], "file_label": "audio"}

        if extra_info:
          metadata = {**metadata, **extra_info}
        if self._concat_rows:
            au = [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=metadata or {}
                )
            ]
        else:
            au = [
                Document(text=text, metadata=metadata or {}) for text in text_list
            ]

        #Load visual content
        file_label_path = VID_LABEL_PATH + "/" + file_label_name + "_img.txt"
        with open(file_label_path, "r") as a:
          text = a.read()

        df = pd.DataFrame(json.loads(text)).transpose()

        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()
        metadata = {"ID": str(uuid.uuid4()), "TITLE": file.name.split('.')[0], "file_label": "visual"}

        if extra_info:
          metadata = {**metadata, **extra_info}
        if self._concat_rows:
            img = [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=metadata or {}
                )
            ]
        else:
            img = [
                Document(text=text, metadata=metadata or {}) for text in text_list
            ]
        return au + img


def manually_construct_nodes_from_video(documents):
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    metadata_chunk = []
    for doc_idx, page in enumerate(documents):
        if page.metadata.get("file_label") == "audio":
            file_name = page.metadata.get("file_name").split(".")[0] + "_au.txt"
            file_label_path = VID_LABEL_PATH + "/" + file_name

            with open(file_label_path, "r") as a:
                text = a.read()
            label = json.loads(text)
            for i in label.keys():
                metadata_chunk.append(label.get(i))
                start_time = label.get(i).get("start_time")
                end_time = label.get(i).get("end_time")

                cur_au_chunks = extract_text_between_times(documents[doc_idx].text, start_time, end_time)
                text_chunks.extend(cur_au_chunks)
                doc_idxs.extend([doc_idx] * len(cur_au_chunks))
        if page.metadata.get("file_label") == "visual":
            file_name = page.metadata.get("file_name").split(".")[0] + "_img.txt"
            file_label_path = VID_LABEL_PATH + "/" + file_name

            with open(file_label_path, "r") as a:
                text = a.read()
            label = json.loads(text)
            for i in label.keys():
                cur_metadata = {"start_time": label.get(i).get("start_time"), "end_time": label.get(i).get("end_time"), "tags": label.get(i).get("description").get("tags") }
                metadata_chunk.append(cur_metadata)
                cur_img_chunks = [label.get(i).get("description").get("caption")]
                text_chunks.extend(cur_img_chunks)
                doc_idxs.extend([doc_idx] * len(cur_img_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        metadata = metadata_chunk[idx]
        src_doc_idx = doc_idxs[idx]
        src_page = documents[src_doc_idx]
        node = TextNode(
            text=text_chunk,
            metadata=metadata,
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=src_page.id_)},
        )

        start_char_idx = src_page.text.find(node.get_content(metadata_mode=MetadataMode.NONE))
        node.metadata.update(src_page.metadata)
        if idx > 0:
            node.relationships[NodeRelationship.PREVIOUS] = nodes[idx - 1].as_related_node_info()
        if idx < len(nodes) - 1:
            node.relationships[NodeRelationship.NEXT] = nodes[idx + 1].as_related_node_info()
        nodes.append(node)
    return nodes

def read_video(vid_path):
    reader = SimpleDirectoryReader(input_dir=vid_path, file_extractor={".mp4": MyVideoReader()})
    documents = reader.load_data()
    for doc in documents:
        doc.excluded_llm_metadata_keys = []
    vid_nodes = manually_construct_nodes_from_video(documents)
    return vid_nodes

class MyPDFReader(BaseReader):
    """PDF parser."""
    def load_data(self, file, extra_info=None):
        with open(file, "rb") as fp:
            # Create a PDF object
            pdf = PdfReader(str(file))

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            docs = []

            for page in range(num_pages):
                    # Extract the text from the page
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]

                    metadata = {"ID": str(uuid.uuid4()), "TITLE": file.name.split('.')[0], "page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)

                    docs.append(Document(text=page_text, metadata=metadata))
        return docs

class MyNewsReader(BaseReader):

    def split_path(self, path):
        # Split the path into directory and filename components
        directory, filename = os.path.split(path)
        # Split the directory into its parent directory and folder name
        parent_directory, folder_name = os.path.split(directory)
        return parent_directory, folder_name, filename

    def load_data(self, file, extra_info=None):
      parent_directory, folder_name, filename = self.split_path(str(file))
      result = []

      if folder_name == "Arirang":
        with open(file, "r") as f:
          json_data = json.load(f)
          for i in range(len(json_data)):
            metadata = {
                'ID': json_data[i].get('artcl_id'),
                'INPUT_TIME': json_data[i].get('input_dtm'),
                'BROADCAST_TIME': json_data[i].get('brdc_schd_dtm'),
                'TITLE': json_data[i].get('artcl_titl'),
                'ARTICLE_TITLE_EN': json_data[i].get('artcl_titl_en'),
                'ARTICLE_TYPE': json_data[i].get('artcl_typ_cd')
                }
            if json_data[i].get('anc_ment_ctt') and json_data[i].get('artcl_ctt'):
              text = json_data[i].get('anc_ment_ctt') + "\n" + json_data[i].get('artcl_ctt')
            else:
              try:
                  text =  json_data[i].get('artcl_ctt')
              except:
                  continue
            if extra_info:
              metadata = {**metadata, **extra_info}
            # load_data returns a list of Document objects
            result.append(Document(text=text, metadata = metadata or {}))
        return result
    
      if folder_name == "TNS":
        with open(file, "r") as f:
          json_data = json.load(f)[0]
          metadata = {
                'ID': json_data.get('STORY_ID'),
                'TITLE': json_data.get('STORY_TITLE'),
                #'Source': json_data.get('Source'),
                #'Link': json_data.get('Link'),
                }
          text = json_data.get('CONTENTS')
          if extra_info:
              metadata = {**metadata, **extra_info}
          # load_data returns a list of Document objects
          result.append(Document(text=text, metadata = metadata or {}))
        return result


def read_document(doc_path, chunk_size=1024, chunk_overlap=10):
    splitter = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc_reader = SimpleDirectoryReader(input_dir=doc_path,  file_extractor={".json": MyNewsReader(), ".pdf": MyPDFReader()}, recursive=True)
    documents = doc_reader.load_data()
    for doc in documents:
        doc.excluded_llm_metadata_keys = []
        doc.text_template = '{metadata_str}\n{content}'
    doc_nodes = splitter.get_nodes_from_documents(documents)
    doc_nodes = sentence_window_node_parser(doc_nodes)
    return doc_nodes