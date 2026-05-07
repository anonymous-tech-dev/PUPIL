from pathlib import Path

MAX_ATTEMPTS = 12

# Paths
BASE_DIR = Path("/home/Pupil")
TRANSCRIPT_DIR = BASE_DIR / "dataset_curation/dataset/transcripts_db/edubench_train_db"
VIDEO_DIR = BASE_DIR / "dataset_curation/dataset/videos_db/train_vids"
OUTPUT_BASE_DIR = BASE_DIR / "dataset_curation/dataset/queries_db/final_train_1k"

# Models
GPT_EVAL_MODEL = "gpt-5.1_2025-11-13"  # High intelligence for validation
GPT_REFINER_MODEL = "gpt-5.1_2025-11-13"   # Fast JSON formatting
MMCT_ENDPOINT = "https://<AZURE_OPENAI_ENDPOINT>"

# --- CONSTANTS ---
# Valid Cognitive Categories for the Nudge
CAT_SYMBOLS = "symbols_ocr"
CAT_SPATIAL = "spatial_ geometric"
CAT_TRANSCRIPT = "transcript_comprehension"
CAT_ACTION = "physical_action"
CAT_FINE_GRAINED = "fine_grained"

# --- VIDEO CATEGORY MAPPING ---
# Logic: If the video filename contains the KEY, use that LIST of categories.

DEFAULT_CATEGORY_MIX = [CAT_SYMBOLS, CAT_ACTION, CAT_SPATIAL, CAT_FINE_GRAINED, CAT_SYMBOLS]

VIDEO_CATEGORY_MAP = {
    # MATH / CHEM / PHYSICS (Heavy Symbols)
    "chem":     [CAT_SYMBOLS, CAT_SYMBOLS, CAT_FINE_GRAINED, CAT_ACTION, CAT_SYMBOLS],
    "math":     [CAT_SYMBOLS, CAT_SYMBOLS, CAT_SPATIAL, CAT_SYMBOLS, CAT_ACTION],
    "howto":  [CAT_SYMBOLS, CAT_ACTION, CAT_SPATIAL, CAT_ACTION],
    "lewin":    [CAT_ACTION, CAT_ACTION, CAT_ACTION, CAT_ACTION, CAT_SPATIAL], # Walter Lewin demos

    # ENGINEERING / ROBOTICS (Heavy Spatial & Action)
    "drawing":  [CAT_SPATIAL, CAT_SPATIAL, CAT_SYMBOLS, CAT_FINE_GRAINED],
    "robotics": [CAT_SPATIAL, CAT_ACTION, CAT_ACTION, CAT_SYMBOLS, CAT_SPATIAL],
    "analog":  [CAT_SYMBOLS, CAT_SPATIAL, CAT_SYMBOLS, CAT_FINE_GRAINED],
    "material": [CAT_FINE_GRAINED, CAT_FINE_GRAINED, CAT_SYMBOLS, CAT_ACTION, CAT_SPATIAL],
    
    # MEDICAL / BIOLOGY (Heavy Fine-Grained)
    "pathology": [CAT_FINE_GRAINED, CAT_FINE_GRAINED, CAT_ACTION, CAT_SYMBOLS, CAT_FINE_GRAINED],
    "bio":       [CAT_FINE_GRAINED, CAT_ACTION, CAT_SPATIAL, CAT_SYMBOLS, CAT_FINE_GRAINED],
}