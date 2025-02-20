
import ast
from datetime import datetime
import json
from typing import Optional


def log_final_results(final_results: dict, stats: Optional[dict], log_final_file: str):
    """
    Logs the final results and statistics to a file with a timestamp.
    Ensures that list values are formatted as bullet points under their respective headings.
    """
    
    with open(log_final_file, "a") as log_file:
        log_file.write(f"\n--- Log Entry: {datetime.now()} ---\n")
        log_file.write("Final Results:\n")

        for key, values in final_results.items():
            log_file.write(f"{key.strip()}:\n")  # Print key heading
            
            print("values:\n", values)
            
            if key == "final_full_summary":
                log_file.write(f"{values.strip()}\n\n")  # Print summary directly
            else:
             # Convert string representation of lists to actual lists
                try:
                    value_list = ast.literal_eval(values.strip())  # Converts to list safely
                    print("value_list", value_list)
                    if isinstance(value_list, list):
                        for item in value_list:
                            log_file.write(f"- {item}\n")  # Print each item as a bullet point
                    else:
                        log_file.write("- None\n")  # Handle unexpected format
                except (ValueError, SyntaxError):
                    log_file.write("- None\n")  # If conversion fails, log as None

            log_file.write("\n")  # Add spacing for readability
        
        if stats:
            log_file.write("Statistics:\n")
            for key, value in stats.items():
                log_file.write(f"{key}: {value}\n")

        log_file.write("--- End of Entry ---\n")


def log_query(prompt, response_msg, cost_for_query, model_name, log_file):
    with open(log_file, "a") as log:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prompt": prompt,
            "response": response_msg,
            "cost_usd": round(cost_for_query, 6),
        }
        log.write(json.dumps(log_entry) + "\n")