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
    hybrid_vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
    keyword_min_token_length = int(os.getenv("KEYWORD_MIN_TOKEN_LENGTH", "2"))
    strict_citation_gate = os.getenv("STRICT_CITATION_GATE", "1") == "1"
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.58"))
    stale_doc_policy = os.getenv("STALE_DOC_POLICY", "warn").strip().lower()
    stale_doc_days = int(os.getenv("STALE_DOC_DAYS", "180"))
    language_primary = os.getenv("LANGUAGE_PRIMARY", "hu").strip().lower()
    language_allowed = [x.strip().lower() for x in os.getenv("LANGUAGE_ALLOWED", "hu,en").split(",") if x.strip()]
    exports_path = Path(os.getenv("EXPORTS_PATH", "./data/exports")).resolve()
    chat_enabled = os.getenv("CHAT_ENABLED", "1") == "1"
    chat_api_token = os.getenv("CHAT_API_TOKEN", "").strip()
    chat_allowed_origins = [x.strip() for x in os.getenv("CHAT_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
    chat_rate_limit_per_min = int(os.getenv("CHAT_RATE_LIMIT_PER_MIN", "60"))

    db_path = Path(os.getenv("DB_PATH", "./helpbot.sqlite3")).resolve()
    docs_path = Path(os.getenv("DOCS_PATH", "./data/docs")).resolve()


settings = Settings()

# Keep policy safe/default even if env has bad value.
if settings.stale_doc_policy not in {"off", "warn", "escalate"}:
    settings.stale_doc_policy = "warn"
