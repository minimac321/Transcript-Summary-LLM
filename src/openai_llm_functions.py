import openai
import tiktoken

from cost_tracker import QueryCostTracker
from constants import OPENAI_MAX_TOKENS_CLASSIFICATION, OPENAI_TEMPERATURE
from log_utils import log_query



def count_tokens_openai(text: str, model: str) -> int:
    """Returns the token count of a given text for a specified model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_openai_cost(messages, response_raw, model_name, openai_pricing_dict: dict) -> float:
    """
    Calculate the estimated cost of an OpenAI API call in dollars.

    Args:
        messages (list): The input messages sent to OpenAI, including roles and content.
        response_raw (dict): The raw response from OpenAI's API call.
        model_name (str): The model used for the API call. Defaults to "gpt-3.5-turbo".

    Returns:
        float: The estimated cost of the query in USD.
    """
    # Tokenizer setup
    model_encodings = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
    }

    if model_name not in openai_pricing_dict or model_name not in model_encodings:
        raise ValueError(f"Unsupported model: {model_name}")

    # Get tokenizer
    tokenizer = tiktoken.get_encoding(model_encodings[model_name])

    # Calculate input tokens
    input_text = " ".join([msg["content"] for msg in messages])
    input_tokens = len(tokenizer.encode(input_text))

    # Calculate output tokens
    output_text = response_raw.choices[0].message.content
    output_tokens = len(tokenizer.encode(output_text))

    # Calculate total cost
    input_cost = (input_tokens / 1000) * openai_pricing_dict[model_name]["input"]
    output_cost = (output_tokens / 1000) * openai_pricing_dict[model_name]["output"]
    total_cost = input_cost + output_cost

    return total_cost


def make_openai_query(
    prompt: str, model_name: str, openai_pricing_dict: dict, 
    cost_tracker: QueryCostTracker,
    log_file: str, write_queries_to_log_file: bool = True,
):
    """
    Make a query to the OpenAI API and return the response.

    Args:
        prompt (str): The prompt for the query.
        model_name (str): The model used for the API call. Defaults to "gpt-3.5-turbo".

    Returns:
        dict: The response from the OpenAI API.
    """
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        # max_tokens=int(OPENAI_MAX_TOKENS_CLASSIFICATION*1.3),
        temperature=OPENAI_TEMPERATURE,
    )
    # Extract response content
    response_msg = response.choices[0].message.content
    # Estimate Cost
    cost_for_query = calculate_openai_cost(messages=messages, response_raw=response, model_name=model_name, openai_pricing_dict=openai_pricing_dict)
    cost_tracker.record_query_cost(cost_for_query)
    
    # Log the query and response
    if write_queries_to_log_file:
        log_query(prompt, response_msg, cost_for_query, model_name, log_file=log_file)
    
    return response_msg, cost_for_query

