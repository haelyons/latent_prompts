from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_LAYERS = 32  # layers in Llama 3.1 8B

# Layer 24 is where SyA and GA disentangle (per paper analysis for 32-layer model)
EXTRACTION_LAYER = 24
DEFAULT_MEASURE_LAYER = 24

MODEL_PROFILES = {
    "8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "hidden_dim": 4096,
        "extraction_layer": 24,
        "candidate_layers": [24],
        "device_map": "cuda",
    },
    "70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "n_layers": 80,
        "hidden_dim": 8192,
        "extraction_layer": 60,
        "candidate_layers": [55, 60, 65],
        "device_map": "auto",  # 2-GPU sharding via accelerate
    },
}

def get_profile(name: str = "8b") -> dict:
    if name not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {name!r}. Choose from {list(MODEL_PROFILES)}")
    return MODEL_PROFILES[name]

# Pairs per dataset (aggregated across 9 datasets via SVD)
N_PAIRS_PER_DATASET = 150

SEED = 42

# Base path to disentangle-sycophancy data
DISENTANGLE_DATA_PATH = Path(__file__).parent / "disentangle-sycophancy" / "data"
FACTORIAL_DATA_DIR = DISENTANGLE_DATA_PATH / "factorial"

FACTORIAL_DATASETS = [
    "math_factorial.json",
    "claims_factorial.json",
    "companies_factorial.json",
    "cities_pos_factorial.json",
    "cities_neg_factorial.json",
    "larger_than_factorial.json",
    "smaller_than_factorial.json",
    "sp_en_trans_factorial.json",
    "counterfactual_factorial.json",
]

# Additional data paths
PRAISE_DATA_PATH = DISENTANGLE_DATA_PATH / "praise.json"
TRUTHFULQA_PATH = DISENTANGLE_DATA_PATH / "truthfulqa.jsonl"


BEHAVIORS = {
    "sya": {
        "name": "Sycophantic Agreement",
        "description": "Model echoes incorrect user claims",
        "label_field": "syc",  # Field name in factorial data
        "filter": {"user_claim_is_correct": False},  # Only when user is wrong
        "positive": "Agrees with factually wrong statement (syc=1)",
        "negative": "Correctly disagrees with wrong statement (syc=0)",
    },
    "ga": {
        "name": "Genuine Agreement", 
        "description": "Model echoes correct user claims",
        "label_field": "ga",
        "filter": {"user_claim_is_correct": True},  # Only when user is correct
        "positive": "Agrees with factually correct statement (ga=1)",
        "negative": "Incorrectly disagrees with correct statement (ga=0)",
    },
    "sypr": {
        "name": "Sycophantic Praise",
        "description": "Model includes excessive flattery",
        "label_field": "pr",
        "filter": None,  # All examples
        "positive": "Response with positive praise (praise_present=True, praise_negated=False)",
        "negative": "Response with neutral/no praise (praise_present=False)",
    },
}

# Mapping from our behavior names to label fields in factorial data
BEHAVIOR_LABEL_MAP = {
    "sya": "syc",
    "ga": "ga",
    "sypr": "pr",
}

DIRECTIONS_DIR = Path("directions")

# Individual direction files pattern: {behavior}_layer{layer}_n{count}_svd.pt
# ex: sya_layer24_n1k3_svd.pt, ga_layer24_n1k3_svd.pt, sypr_layer24_n1k3_svd.pt

VALIDATION_SPLIT = 0.2  # 20% holdout for AUROC validation
MIN_AUROC_THRESHOLD = 0.6  # warn if direction AUROC is below this
VALIDATION_SAMPLES = 500  # number of samples for AUROC computation

# maybe a bit low here?
MAX_NEW_TOKENS = 100

SCORE_THRESHOLDS = {
    "high": 0.3,      # Strong alignment with direction
    "moderate": 0.1,  # Moderate alignment
    "neutral": -0.1,  # Neutral range
    # Below neutral = opposite of behavior
}
