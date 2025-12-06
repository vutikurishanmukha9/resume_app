# config.py
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_CACHE_FILE = BASE_DIR / "models" / "job_embeddings_cache.pkl"
EMBEDDING_HASH_FILE = BASE_DIR / "models" / "job_embeddings_hash.txt"
JOB_DATA_CSV = BASE_DIR / "data" / "job_title_des.csv"
MODEL_DIR = BASE_DIR / "models"
MAX_TEXT_LENGTH = 5000
MAX_PDF_PAGES = 5
TOP_MATCHES = 3
MIN_TEXT_LENGTH = 50
