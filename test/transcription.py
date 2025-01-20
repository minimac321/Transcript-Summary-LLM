import os
import openai
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Initialize API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to transcribe audio
def transcribe_audio(file_path: str, model: str = "whisper-1") -> str:
    """Transcribe audio from a given file."""
    try:
        # Open audio file
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model=model,  # OpenAI Whisper model
                file=audio_file
            )
        return response['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

# Function to correct grammar in transcription
def correct_grammar(transcript: str, model: str = "gpt-4", temperature: float = 0.7) -> str:
    """Correct grammar and spelling errors in the transcript."""
    try:
        system_prompt = """
        You are a helpful AI assistant. Correct any grammar, spelling, or formatting issues in the provided text.
        Ensure the output is clear and readable while preserving the original meaning.
        """
        # Generate response using OpenAI Chat API
        response = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error correcting grammar: {e}")
        return transcript

# Function to process an audio file
def process_audio_file(file_path: str, output_dir: str = "outputs"):
    """Process the audio file: transcribe, correct grammar, and save output."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Transcribe the audio
        print(f"Transcribing audio file: {file_path}")
        transcript = transcribe_audio(file_path)

        # Perform grammar correction
        print("Correcting grammar...")
        corrected_transcript = correct_grammar(transcript)

        # Save the corrected transcript
        def write_file(output_file, transcript):
            with open(output_file, "w") as f:
                f.write(transcript)

        base_name_uncorrected = f"{os.path.basename(file_path)}_uncorrected.txt"
        write_file(output_file=base_name_uncorrected, transcript=transcript)
        print(f"Uncorrected Transcript saved at: {base_name_uncorrected}")

        base_name = f"{os.path.basename(file_path)}.txt"
        write_file(output_file=base_name, transcript=corrected_transcript)
        print(f"Transcript saved at: {base_name}")

    except Exception as e:
        print(f"Error processing audio file: {e}")

# Main function
def main():
    # Input file path
    file_path = input("Enter the path to the audio file (e.g., 'uploads/audio_file.wav'): ").strip()

    # Check if the file exists
    if not os.path.exists(file_path):
        print("File not found. Please provide a valid file path.")
        return
    else:
        print(f"File Found")

    # Process the audio file
    # process_audio_file(file_path)

if __name__ == "__main__":
    main()