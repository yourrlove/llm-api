from flask import Flask
from flask_cors import CORS
import os

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(PROJECT_PATH, "database/data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "llm_analyze_data")
INDEX_MAPPING = {
    "doc-index": "document",
    "img-index": "image",
    "au-index": "audio",
    "vid-index": "video",
}

def create_app():
    app = Flask(__name__)

    if os.environ.get('FLASK_ENV') == 'development':
        CORS(app)

    from app.api.routes import api_blueprint
    app.register_blueprint(api_blueprint)

    return app