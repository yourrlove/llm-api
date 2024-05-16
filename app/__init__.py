from flask import Flask
from flask_cors import CORS
import os

PROJECT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def create_app():
    app = Flask(__name__)

    if os.environ.get('FLASK_ENV') == 'development':
        CORS(app)

    from app.api.routes import api_blueprint
    app.register_blueprint(api_blueprint)

    return app