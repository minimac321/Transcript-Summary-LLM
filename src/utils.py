from typing import List

from constants import GEMINI_MODEL_NAMES_LIST, MODEL_TOKEN_LIMITS, OPENAI_MODEL_NAMES_LIST
from gemini_llm_functions import count_tokens_gemini
from openai_llm_functions import count_tokens_openai


def count_tokens_func(text: str, model: str) -> int:
    """Returns the token count of a given text for a specified model."""
    if model in GEMINI_MODEL_NAMES_LIST:
        return count_tokens_gemini(text, model)
    elif model in OPENAI_MODEL_NAMES_LIST:
        return count_tokens_openai(text, model)
    else:
        print("(ERROR) - Model is not part of OpenAI or Gemini models")
        assert False


def split_text_into_chunks(text: str, model: str) -> List[str]:
    """Splits text into chunks based on token limits for the given model."""
    token_limit = MODEL_TOKEN_LIMITS.get(model, 10000)  # Default to 10k if unknown
    words = text.split()  # Split by words (not perfect, but avoids cutting mid-sentence)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = count_tokens_func(word, model)  # Estimate token count per word
        if current_tokens + word_tokens > token_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(word)
        current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



        
        
# def consolidate_mappings(mappings: List[dict]) -> dict:
#     """Consolidates multiple speaker mappings by weighting those with more tokens higher."""
#     speaker_roles = {}

#     for mapping in mappings:
#         for speaker, role in mapping.items():
#             if speaker not in speaker_roles:
#                 speaker_roles[speaker] = []
#             speaker_roles[speaker].append(role)

#     final_mapping = {}
#     for speaker, roles in speaker_roles.items():
#         # Weight roles based on occurrence
#         role_counts = Counter(roles)
#         most_common_role = role_counts.most_common(1)[0][0]
#         final_mapping[speaker] = most_common_role

#     return final_mapping

# def parse_speaker_mapping(classification_output: dict, chunk: list) -> dict:
#     """
#     Parses the output of the 'Classify Roles' tool into a speaker mapping dictionary.

#     Parameters:
#     - classification_output (dict): The output from the "Classify Roles" tool (e.g., {'spk_0': 'financial planner', 'spk_1': 'client'}).
#     - chunk (list): The list of transcript segments, where each segment contains 'speaker_label' and 'transcript'.

#     Returns:
#     - dict: A dictionary mapping speaker IDs (e.g., "spk_0") to their roles (e.g., "financial planner", "client").
#     """
#     if not isinstance(classification_output, dict):
#         raise ValueError("Invalid classification output format. Expected a dictionary.")

#     speaker_mapping = {}
#     for segment in chunk:
#         speaker = segment['speaker_label']  # e.g., "spk_0", "spk_1"
#         if speaker in classification_output:
#             role = classification_output[speaker]
#             if speaker in speaker_mapping:
#                 # Check for role conflicts
#                 if speaker_mapping[speaker] != role:
#                     print(f"Warning: Speaker '{speaker}' was previously assigned as '{speaker_mapping[speaker]}' and is now being reassigned to '{role}'.")
#             else:
#                 speaker_mapping[speaker] = role
#         else:
#             print(f"Warning: Speaker '{speaker}' not found in classification output.")

#     return speaker_mapping