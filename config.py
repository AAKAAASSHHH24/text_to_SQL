import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DATABASE_PATH = "chinook.db"
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0
    
    # Chart configuration
    CHART_HEIGHT = 400
    MAX_PIE_SLICES = 10
    MAX_BAR_ITEMS = 20