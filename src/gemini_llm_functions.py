
import enum
import time
from typing import Dict
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from pydantic import BaseModel
from vertexai.preview.generative_models import GenerativeModel

from cost_tracker import QueryCostTracker
from constants import GEMINI_TEMPERATURE
from log_utils import log_query


def count_tokens_gemini(text: str, model: str) -> int:
    """Returns the token count of a given text for a specified model."""
    model = GenerativeModel(model)
    response = model.count_tokens(text)
    return response.total_tokens


def calculate_gemini_cost(prompt, response, model_name, gemini_pricing_dict: dict):
    """
    Calculate the cost of a Gemini API call based on the model used.

    Args:
        prompt (str): The input prompt.
        response (str): The generated response.
        model_name (str): The name of the Gemini model used.

    Returns:
        float: The estimated cost of the API call.
    """
    input_tokens = len(prompt.split())
    output_tokens = len(response.split())

    if model_name not in gemini_pricing_dict:
        raise ValueError(f"Unknown model: {model_name}")

    model_pricing = gemini_pricing_dict[model_name]
    input_cost = (input_tokens / 1000000) * model_pricing["input"]
    output_cost = (output_tokens / 1000000) * model_pricing["output"]

    return input_cost + output_cost


class SpeakerLabel(enum.Enum):
    FINANCIAL_PLANNER = "financial planner"
    CLIENT = "client"
    
# Pydantic model to enforce structured representation
class SpeakerClassification(BaseModel):
    speakers: Dict[str, SpeakerLabel]  
    
# # Define a key-value structure instead of Dict
# class SpeakerAssignment(BaseModel):
#     speaker: str  # e.g., "spk_0"
#     role: SpeakerLabel
#     # role: Literal[SpeakerLabel.FINANCIAL_PLANNER, SpeakerLabel.CLIENT]

# class SpeakerClassification(BaseModel):
#     speakers: List[SpeakerAssignment]  # List of objects instead of Dict


def make_gemini_query(
    prompt, model_name: str, gemini_pricing_dict: dict, 
    cost_tracker: QueryCostTracker,
    log_file: str, write_queries_to_log_file: bool = True,
    max_retries: int = 5
):
    """
    Make a query to the Gemini API and return the response.

    Args:
        prompt (str): The prompt for the query.
        model_name (str): The model used for the API call. Defaults to "gemini-pro".

    Returns:
        tuple: The response from the Gemini API and the estimated cost.
    """
    model = genai.GenerativeModel(model_name)
    
    # safety_settings = {
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    # }
    
    # Define the JSON schema to validate the response
    
    # Configure the generation settings
    generation_config = genai.types.GenerationConfig(
        temperature=GEMINI_TEMPERATURE,
        # max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,  # Uncomment and set if needed:
    )
    wait_time = 30
    
    for attempt in range(max_retries):
        try:
            # Call the API to generate the response
            response = model.generate_content(
                contents=prompt,   
                generation_config=generation_config,
                # safety_settings=safety_settings # Uncomment if using safety settings:
            )

            response_text = response.text
            cost_for_query = calculate_gemini_cost(prompt, response_text, model_name=model_name, gemini_pricing_dict=gemini_pricing_dict)
            cost_tracker.record_query_cost(cost_for_query)
            
            # Log the query and response
            if write_queries_to_log_file:
                log_query(prompt, response_text, cost_for_query, model_name, log_file=log_file)
            
            return response_text, cost_for_query
        
        except TooManyRequests:
            print(f"429 Error: Too many requests. Retry attempt {attempt} in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 1
        except ResourceExhausted:
            print(f"Error: Resource Exhausted. Retry attempt {attempt} in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 1
        
    print("Max retries reached. Exiting.")
    exit()
    return None