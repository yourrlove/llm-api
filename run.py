from app import create_app
from os.path import join, dirname
from dotenv import load_dotenv
import nest_asyncio
from pyngrok import ngrok

dotenv_path = join(dirname(__file__), 'enviroment.env')
load_dotenv(dotenv_path)

app = create_app()

if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)
    
    # Apply nest_asyncio to avoid event loop issues
    nest_asyncio.apply()
    app.run(host="0.0.0.0", port=5000)