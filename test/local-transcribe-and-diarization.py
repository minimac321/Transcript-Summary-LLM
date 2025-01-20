'''
Installations:
# Install Whisper for transcription
pip install openai-whisper ffmpeg

# Install PyTorch with MPS (Metal backend) support
pip install torch torchvision torchaudio

# Install Pyannote.audio for speaker diarization
pip install pyannote.audio
pip install librosa
'''

import os
from pydub import AudioSegment
from pyannote.audio.pipelines import SpeakerDiarization
from dotenv import load_dotenv
import whisper
import torch
from tqdm import tqdm

# print(os.getcwd())
# Load environment variables
load_dotenv("app/.env")
HUGGINGFACE_TOKEN = os.getenv('HUGGING_FACE_API_KEY')

# Device setup for Apple Silicon (M2)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"  # Force CPU execution to avoid MPS errors
print(f"device: {device}")


# Load Whisper model (optimized for M2 Mac)
whisper_model = whisper.load_model("base", device="cpu", download_root="./models")  # Options: 'tiny', 'base', 'small', 'medium', 'large'

# Load Pyannote model for speaker diarization
pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Function to perform diarization
def diarize_audio(file_path):
    diarization = pipeline(file_path)
    segments = []
    unique_speakers = list()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
        if speaker not in unique_speakers:
            unique_speakers.append(speaker)
    return segments, unique_speakers

# Function to transcribe audio using Whisper
def transcribe_audio(file_path: str, temperature=None, verbose=False):
    result = whisper_model.transcribe(
        audio=file_path,
        verbose=verbose,
        # temperature=temperature,
    )
    return result['text']

# Function to process audio with or without diarization
def process_audio(file_path, output_file, use_diarization=True):
    results = []

    if use_diarization:
        # Perform speaker diarization
        print("Performing speaker diarization...")
        segments, unique_speakers = diarize_audio(file_path)
        print(f"After Diarizing Audio, found {len(unique_speakers)} unique speakers and {len(segments)} audio segments. Speakers: {unique_speakers[:5]}")

        # Process each speaker segment
        print("Transcribing each speaker segment...")
        audio = AudioSegment.from_file(file_path)
        for segment in tqdm(segments):
            start, end = segment['start'], segment['end']
            speaker = segment['speaker']
            
            # Extract segment
            clip = audio[start * 1000:end * 1000]  # milliseconds
            clip.export("temp.wav", format="wav")

            # Transcribe the segment
            text = transcribe_audio(file_path="temp.wav")
            results.append(f"{speaker}: {text}")
            print(f"{speaker}: {text}")
    else:
        # Transcribe full audio without diarization
        print("Transcribing full audio...")
        text = transcribe_audio(file_path=file_path)
        results.append(f"Speaker 1: {text}")  # Assume single speaker or unknown speaker

    # Save results to output file
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    print(f"Transcription saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    # File paths
    input_dir = "uploads"
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)  # Ensure output directory exists

    # input_file = os.path.join(input_dir, "Sample1/test1.wav")
    input_file = os.path.join(input_dir, "Sample2/Neuroscientist-caffeine.wav")
    # input_file = os.path.join(input_dir, "Sample3/panel-discussion.wav")
    output_file = os.path.join(outputs_dir, "output_transcription-s2-w-diar.txt")

    # User choice: Use diarization or not
    use_diarization = True  # Set to False to skip diarization and transcribe raw audio

    # Process audio based on flag
    process_audio(input_file, output_file, use_diarization=use_diarization)