This is the repository for Demo Rest Api Custom LLM

<h3> API </h3>

1. geminisoftvn.ddns.net:9000/ai/v1/llm/assets/<query>
<ul>
    <li>Using to retrieve assets (such as image, audio, video, pdf or news) or the ormation in any file</li>
    <li>Example query:
    <ul>
        <li>Give me some images about panda</li>
        <li>Recommend a video mention how to make a polite request</li>
        <li>벚꽃 이미지 좀 보여주세요</li>
        <li>Give me the news about the 47th prime minister of the country of Korea.</li>
        <li>What is negative rejection?</li>
    </ul>
    </li>
</ul>

2. geminisoftvn.ddns.net:9000/ai/v1/llm/articles/<query>
<ul>
    <li>Using to generate content from News in db</li>
    <li>Example query:
        <ul>
            <li>Generate a 250 word long paragraph to describe South Korea in Covid pandemic.</li>
            <li>Write a 500-word essay and discuss the following keywords: daily life, COVID-19, and Korea.</li>
            <li>500자 분량의 에세이를 작성하고 일상생활, 코로나19, 한국이라는 키워드에 대해 토론하세요.</li>
        </ul>
    </li>
</ul>

3. geminisoftvn.ddns.net:9000/ai/v1/llm/data?file_path&db_name
<ul>
    <li>Using to upload data into VectorDB (Data must be set up like folder structure)</li>
    <li>file_path: data/Images, data/Audios, data/Videos, data/Documents</li>
    <li>db_name: LlamaIndex_img_index, LlamaIndex_au_index, LlamaIndex_vid_index, LlamaIndex_doc_index</li>
    <li></li>
</ul>

<h3>Flask App Structure</h3>
Inside that directory, it will generate the initial project structure :

```
API_llm/
├──enviroment.env
├── app
│   ├── __init__.py
│   ├── api
│   │   ├── __init__.py 
│   │   └── routes.py 
│   ├── templates
│   │   └── form.html   
│   ├── models
│   │   ├── model
│   │   └── mistral-7b-instruct-v0.2.Q4_K_M-001.gguf
│   ├── services
│   │   ├── retrive_assets.py
│   │   └── generate_text.py
│   ├── utils     
│   │   ├── data_reader.py
│   │   ├── node_postprocessors.py 
│   │   └── prompt_templates.py
│   ├── database
│   │   └── index.py
│   ├── data
│   │   ├── Images
│   │   │   ├── {name}.jpg
│   │   │   └── label
│   │   │      └── {name}.txt
│   │   ├── Audios
│   │   │   ├── {name}.mp3
│   │   │   └── label
│   │   │      ├── {name}.srt
│   │   │      └── {name}.txt
│   │   ├── Videos
│   │   │   ├── {name}.mp4
│   │   │   └── label
│   │   │      ├── {name}.srt
│   │   │      ├── {name}_au.txt
│   │   │      └── {name}_img.txt
│   │   ├── Documents
│   │   │   ├──News
│   │   │   │   ├── Airang
│   │   │   │   │   └── {name}.json
│   │   │   │   └── TNS
│   │   │   │       └── {name}.json
│   │   │   └──pdf
│   │   │       └── {name}.pdf
├── requirements.txt  
└── run.py

```
# Demo in Command Prompt

<h3>Set up the enviroment </h3>

```
cd /home/geminisoft/workdir/llm/api_llm

```

<h3>Create Anaconda Virtual Environment And Install Packages </h3>

```
#conda create -n rag python=3.10

conda activate rag

pip install -r requirements.txt

```

<h3>Run</h3>

```
python run.py

```
