import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from datetime import datetime
from dotenv import load_dotenv


# Constants
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_MAX_OUTPUT_TOKENS = 1000
GEMINI_TEMPERATURE = 0.7
LLM_LOG_FILE_NAME = "gemini_api_log.txt"


import sys
current_directory = os.getcwd()
sys.path.append(current_directory)
app_dir = os.path.join(current_directory, "app")
sys.path.append(app_dir)

# Load .env variables
load_dotenv("app/.env")


# Define pricing for different models (per 1M tokens)
gemini_pricing_dict = {
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-2.0-flash": {"input": 0.00010, "output": 0.00040},
    "gemini-1.5-flash": {"input": 0.00010, "output": 0.00040}
}


def log_query(prompt, response, cost, model_name, log_file):
    """
    Log the query, response, and cost to a file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Response: {response}\n")
        f.write(f"Estimated Cost: ${cost:.6f}\n")
        f.write("-" * 50 + "\n")


# Example usage
if __name__ == "__main__":
    
    prompt = "Explain the concept of quantum computing in simple terms."
    selected_model_name = "gemini-2.0-flash"
    response, cost = make_gemini_query(prompt, model_name=selected_model_name, gemini_pricing_dict=gemini_pricing_dict)
    print(f"Response: {response.text}")
    print(f"Estimated Cost: ${cost:.6f}")
