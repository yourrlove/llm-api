This is the repository for Demo Rest Api Custom LLM

### API Endpoints

#### 1. Retrieving Assets
- **Endpoint**: `geminisoftvn.ddns.net:9000/ai/v1/llm/assets/<query>`
- **Purpose**: Retrieves assets such as images, audio, video, PDFs, or news based on the provided query.
- **Examples**:
  - "Give me some images about panda"
  - "Recommend a video on how to make a polite request"
  - "벚꽃 이미지 좀 보여주세요" (Korean for "Show me some cherry blossom images")

#### 2. Generating Articles
- **Endpoint**: `geminisoftvn.ddns.net:9000/ai/v1/llm/articles/<query>`
- **Purpose**: Generates content from news stored in the database based on the query.
- **Examples**:
  - "Generate a 250-word long paragraph to describe South Korea in the Covid pandemic."
  - "Write a 500-word essay discussing daily life, COVID-19, and Korea."
  - "500자 분량의 에세이를 작성하고 일상생활, 코로나19, 한국이라는 키워드에 대해 토론하세요." (Korean version of the above query)

#### 3. Data Upload to VectorDB
- **Endpoint**: `geminisoftvn.ddns.net:9000/ai/v1/llm/data?file_path`
- **Purpose**: Uploads data into VectorDB using specified file paths.
- **Details**:
  - `file_path`: Specifies the location of data files (All files are placed in one folder, classified by names starting with document, audio, video, image)
  

### Project Structure (API_llm)

#### Directory Structure
```
API_llm/
├── environment.env
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── templates/
│   │   └── form.html
│   ├── models/
│   │   └── model.py
│   ├── services/
│   │   ├── retrieve_assets.py
│   │   └── generate_text.py
│   ├── utils/
│   │   ├── data_reader.py
│   │   ├── node_postprocessors.py
│   │   └── prompt_templates.py
│   ├── database/
│   │   ├── data/
│   │   │   └── llm_analyze_data/
│   │   └── index.py
├── requirements.txt
└── run.py
```

#### Components
- **environment.env**: Configuration file for environment variables (e.g., API keys).
- **app/**: Main application directory.
  - **api/**: Contains API route definitions (`routes.py`).
  - **templates/**: Stores HTML templates (`form.html`).
  - **models/**: Contains application data models (`model.py`).
  - **services/**: Handles specific business logic (e.g., asset retrieval, text generation).
  - **utils/**: Utility functions used across the application.
  - **database/**: Stores various data files categorized under `data/` and database indexing in `index.py`.

### Running the Project

#### Setup and Execution
1. Navigate to the project directory:
   ```
   cd /home/geminisoft/workdir/llm/api_llm
   ```

2. Create and activate a virtual environment (assuming Anaconda):
   ```
   conda create -n rag python=3.10
   conda activate rag
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   python run.py
   ```

### Additional Notes
- Ensure to replace `open_api_keys` in `.env` file with actual API keys before running the project to handle any necessary authentication or access control.

This structure provides a clear separation of concerns with dedicated directories for routes, models, services, utilities, and database management, making it modular and easier to maintain. Adjustments may be needed based on specific deployment environments or additional functionality requirements.
