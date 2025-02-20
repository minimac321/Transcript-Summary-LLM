# General Constants
USD_TO_ZAR_CONVERSION = 19.10
DEFAULT_FP_NAME = "Morgan"


# Define token limits per model
classify_roles_query_token_count = 500
MODEL_TOKEN_LIMITS = {
    "gpt-4": 8192 - classify_roles_query_token_count,  # Adjust as per your account limits
    "gpt-3.5-turbo": 16000 - classify_roles_query_token_count,  # Adjust as per your account limits
    "gemini-pro": 32768 - classify_roles_query_token_count,
    "gemini-1.5-flash": 1048576 - classify_roles_query_token_count,
    "gemini-2.0-flash": 1048576 - classify_roles_query_token_count,
}


# Google Cloud Platform
GCP_PROJECT_ID = "transcription-fs-app"
GCP_LOCATION = "us-central1"

# AWS default settings
DEFAULT_AWS_REGION_NAME = "us-west-2"
DEFAULT_AWS_BUCKET_NAME = "digital-resume-s3"
# AWS Transcribe constants
MEDIA_FORMAT = "wav"  # Options: "wav", "mp3", "mp4", "flac"
LANGUAGE_CODE = "en-ZA"  # Options: "en-US", "en-GB", "es-US", "fr-CA", etc.
DEFAULT_TRANSCRIBE_MAX_SPEAKERS = 3

VOCABULARY_DIR = "vocabulary"
TRANSCRIBE_VOCABULARY_FILE = f"{VOCABULARY_DIR}/FP-SA-Vocab-V1.csv"


# OpenAI Constants
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"  # Options: "gpt-3.5-turbo", "text-davinci-003", "text-curie-001", "gpt-4"
OPENAI_MODEL_NAMES_LIST = ["gpt-3.5-turbo", "text-davinci-003", "text-curie-001", "gpt-4"]

OPENAI_TRANSCRIPTION_PROJ_ID = "proj_Q8SXL9dUJqv3kBv3PUlotdt6"
OPENAI_ORG_NAME = "org-3m84stKBZ07lg0LvLzHGh5Yb"
OPENAI_MAX_TOKENS_SUMMARY = 300
OPENAI_MAX_TOKENS_EXTRACTION = 300
OPENAI_MAX_TOKENS_CLASSIFICATION = 500
OPENAI_TEMPERATURE = 0.3  # Options: Float between 0 (deterministic) and 1 (creative)


# Pricing per 1,000 tokens (adjust these based on OpenAI's pricing structure)
OPENAI_PRICING_DICT = {
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.0020},  # USD per 1,000 tokens
    "gpt-4": {"input": 0.03, "output": 0.06},             # USD per 1,000 tokens
}


# Google Constants
GEMINI_MODEL_NAMES_LIST = ["gemini-2.0-flash",  "gemini-1.5-flash", "gemini-pro"]
STANDARD_GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_OUTPUT_TOKENS = 10000

# Define pricing for different models (per 1M tokens)
GEMINI_PRICING_DICT = {
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-2.0-flash": {"input": 0.00010, "output": 0.00040},
    "gemini-1.5-flash": {"input": 0.00010, "output": 0.00040}
}