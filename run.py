from app import create_app
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), 'enviroment.env')
load_dotenv(dotenv_path)

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)