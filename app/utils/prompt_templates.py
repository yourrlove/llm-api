IMG_PROMPT_TMPL = """\
The information of some images is given below, each image has information about its main content and some file metadata:
{context_str}

User query:
{query_str}

The user want to ask for a specific information in some images or want to get the main content of some images related to a certain topic. \
Answer the user query based on the given information. \
Use the following format to generate response:
- Answer: The response to the user query.
- IDs: The ids of the images that answer the question.

The response must comply with the format and must not be returned ambiguous or missing elements in the format.
The source must be quoted correctly from the answer source, do not arbitrarily cite other images or documents
The response must contain only information about image, (not audio, video or text file).
Ignore the information that unrelated to the user query and don't include it in the response.
If you cannot find any images related to the query, just say that you don't know, don't try to make up an answer.
It is important that the response must be generated based on the given information only.
If you cannot find the answer in the given information, just say that you don't know, don't try answer using external knowledge.
"""


AU_PROMPT_TMPL = """\
The information of some audios is given below, each audio has information about its main content and some file metadata:
{context_str}

User query:
{query_str}

The user want to ask for a specific information in some audios or want to get the main content of some audios related to a certain topic. \
Answer the user query based on the given information. \
Use the following format to generate response:
- Answer: The response to the user query.
- ID: The ids of the audios that answer the question.
- Time: from "start_time" to "end_time" (the time frame of the event that answer the user's query)

The response must comply with the format and must not be returned ambiguous or missing elements in the format.
The source must be quoted correctly from the answer source, do not arbitrarily cite other audios or documents
The response must contain only information about audio, (not image, video or text file).
Ignore the information that unrelated to the user query and don't include it in the response.
If you cannot find any audios related to the query, just say that you don't know, don't try to make up an answer.
It is important that the response must be generated based on the given information only.
If you cannot find the answer in the given information, just say that you don't know, don't try answer using external knowledge.
"""


VID_PROMPT_TMPL = """\
The information of some videos is given below, each video has information about its main content and some file metadata:
{context_str}

User query:
{query_str}

The user want to ask for a specific information in some videos or want to get the main content of some videos related to a certain topic. \
Answer the user query based on the given information. \
Use the following format to generate response:
- Answer: The response to the user query.
- ID: The ids of the videos that answer the question.
- Time: from "start_time" to "end_time" (the time frame of the event that answer the user's query)

The response must comply with the format and must not be returned ambiguous or missing elements in the format.
The source must be quoted correctly from the answer source, do not arbitrarily cite other videos or documents
The response must contain only information about video, (not image, audio or text file).
Ignore the information that unrelated to the user query and don't include it in the response.
If you cannot find any videos related to the query, just say that you don't know, don't try to make up an answer.
It is important that the response must be generated based on the given information only.
If you cannot find the answer in the given information, just say that you don't know, don't try answer using external knowledge.
"""


DOC_PROMPT_TMPL = """\
Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). \
Ignore irrelevant search results and do not contain it in the answer. \
If there are no search results related to the question, just say that you don't know, don't try to make up an answer and don't use external knowledge. \

Use the following format to generate response:
- Answer: The response to the question.
- ID: The ids of the file that answer the question.
- Title: The title of the file that answer the quesion.

Search results:
{context_str}

Question: {query_str}
"""


QUERY_REWRITE_PROMPT_TMPL = """\
Given a query
If the query is written in Korean, add the phrase '답은 한국어로 작성해야 합니다.' to the end of the query.
If the query is written in English, keep it unchanged.

Query: {query_str}
Rewritten query:
"""


CONTENT_GEN_PROMPT_TMPL = """\
### Context information from multiple sources is below:
------------------------------------------------------
{context_str}

Given the information from multiple sources and not prior knowledge, and a query from a user.
Generate content according to user's query based on the given context.
Ignore irrelevant information in the context and do not include it in the response.
The length of the generated content must be around the word limit number provided by the user.

### User query: {query_str}
### Generated text:
"""