from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    model_provider = os.getenv("MODEL_PROVIDER", "ollama")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    embed_model = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

    top_k = int(os.getenv("TOP_K", "5"))
    min_similarity = float(os.getenv("MIN_SIMILARITY", "0.42"))

    db_path = Path(os.getenv("DB_PATH", "./helpbot.sqlite3")).resolve()
    docs_path = Path("./data/docs").resolve()


settings = Settings()
