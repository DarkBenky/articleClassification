"""Shared paths and model settings for the article-classification pipeline.

All scripts resolve assets through this module so they behave the same
regardless of the current working directory. Large datasets stay on the
external 2 TB drive; the repository only holds code, inputs, and artifacts.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# External, large datasets shared by preprocessing, tokenization, and training.
DATA_ROOT = Path("/media/user/2TB")
CORPUS_PATH = DATA_ROOT / "preprocessed_data.txt"
TOKENIZED_DIR = DATA_ROOT / "tokenizedtext"
RAW_ARTICLES_DIR = DATA_ROOT / "raw_articles"
HF_CACHE_DIR = DATA_ROOT / "huggingface_cache"

# Repository inputs.
CAMEO_CODES_PATH = RAW_DATA_DIR / "CODES.csv"
FIPS_CODES_PATH = RAW_DATA_DIR / "fipsCodes.csv"
NEWS_CATEGORY_PATH = RAW_DATA_DIR / "NEWS_CATEGORY.json"
RUSSIAN_INVASION_PATH = RAW_DATA_DIR / "RussianInvasion.csv"

# Generated label mappings.
UNIQUE_CATEGORIES_PATH = PROCESSED_DATA_DIR / "unique_categories.json"
UNIQUE_LOCATIONS_PATH = PROCESSED_DATA_DIR / "unique_locations.json"
LOCATION_TO_FIPS_PATH = PROCESSED_DATA_DIR / "location_to_fips.json"
UNIQUE_FIPS_LOCATIONS_PATH = PROCESSED_DATA_DIR / "unique_fips_locations.json"

MODEL_PATH = MODELS_DIR / "location_model_best.keras"

CONTEXT_SIZE = 512
TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
