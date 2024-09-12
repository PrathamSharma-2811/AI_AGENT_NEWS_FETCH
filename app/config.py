import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    EXTRACT_KEY = os.getenv("extract__key")
    NEWS_API_KEY = os.getenv("news_api")
    HF_TOKEN = os.getenv("hf_token")
    GEMINI_KEY = os.getenv("gemini_key")
    MONGO_URI = os.getenv('MONGO_URI')
    
