
import os
from typing import Tuple
import boto3
import json
import time
from datetime import datetime

from constants import DEFAULT_AWS_BUCKET_NAME, DEFAULT_AWS_REGION_NAME, DEFAULT_TRANSCRIBE_MAX_SPEAKERS, LANGUAGE_CODE, MEDIA_FORMAT, TRANSCRIBE_VOCABULARY_FILE

class Boto3Client:
    """Manages Boto3 client connections."""
    def __init__(self, 
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region: str = DEFAULT_AWS_REGION_NAME,
        ):
        self.transcribe_client : boto3.client = boto3.client(
            "transcribe",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

class S3Manager:
    """Handles S3 operations."""
    def __init__(self, boto3_client: Boto3Client, bucket_name: str = DEFAULT_AWS_BUCKET_NAME):
        self.s3_client = boto3_client.s3_client
        self.bucket_name = bucket_name

    def upload_file(self, local_file_path: str, s3_key: str) -> str:
        """Uploads a file to S3 and returns the S3 URI."""
        self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
        return f"s3://{self.bucket_name}/{s3_key}"

class TranscriptionManager:
    """Manages transcription jobs."""
    def __init__(self, boto3_client: Boto3Client):
        self.transcribe_client = boto3_client.transcribe_client
        
    def get_client(self) -> boto3.client:
        return self.transcribe_client

    def wait_for_transcription_completion(self, job_name: str):
        """Waits until the transcription job completes."""
        while True:
            response = self.transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]
            if status in ["COMPLETED", "FAILED"]:
                break
            print("Waiting for transcription to complete...Wait 30 seconds")
            time.sleep(30)

        if status == "FAILED":
            raise RuntimeError(f"Transcription job failed: {response}")
        return response

    def fetch_transcription_output(self, job_name: str, s3_output_folder: str) -> dict:
        """Fetches the transcription output JSON."""
        response = self.transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )
        uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
        file_name = s3_output_folder + uri.split("/")[-1]
        print("file_name", file_name)
        transcription_file_data = boto3.client("s3").get_object(
            Bucket=DEFAULT_AWS_BUCKET_NAME, 
            Key=file_name,
        )
        transcription_data = json.loads(transcription_file_data["Body"].read())['results']
        return transcription_data
    
    
    
def initialize_clients(aws_access_key_id: str, aws_secret_access_key: str):
    """Initializes S3 and Transcription clients."""
    boto3_client = Boto3Client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    s3_manager = S3Manager(boto3_client)
    transcription_manager = TranscriptionManager(boto3_client)
    return boto3_client, s3_manager, transcription_manager



def upload_audio_to_s3(s3_manager, local_file_path: str, directory_name: str = "other/audio-files") -> Tuple[str, str]:
    """
    Uploads an audio file to S3 and creates a structured folder hierarchy on S3.

    Parameters:
    - s3_manager: An S3 manager object with `s3_client` and `bucket_name`.
    - local_file_path (str): The path of the local file to upload.
    - directory_name (str): The base directory path inside the S3 bucket. Default is "other/audio-files".

    Returns:
    - str: The S3 URL of the uploaded file.
    """
    # Get the current date in dd-mm-yy format
    current_datetime = datetime.now().strftime("%H%M_%d-%m-%y")
    job_folder = f"job_{current_datetime}"

    # Define S3 paths
    input_s3_key = f"{directory_name}/{job_folder}/input/{os.path.basename(local_file_path)}"
    output_s3_folder = f"{directory_name}/{job_folder}/output/"

    s3_client = s3_manager.s3_client

    # Check if the file already exists in S3
    try:
        s3_client.head_object(Bucket=s3_manager.bucket_name, Key=input_s3_key)
        print(f"File already exists in S3: s3://{s3_manager.bucket_name}/{input_s3_key}")
        return f"s3://{s3_manager.bucket_name}/{input_s3_key}"
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"File does not exist in S3. Proceeding with upload.")
            # pass
        else:
            print(f"Error checking file in S3: {e}")
            raise

    # Upload the file to the input directory on S3
    try:
        s3_client.upload_file(local_file_path, s3_manager.bucket_name, input_s3_key)
        print(f"Uploaded to S3: s3://{s3_manager.bucket_name}/{input_s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise

    # Create the output folder on S3 (S3 "folders" are just keys that end with '/')
    try:
        s3_client.put_object(Bucket=s3_manager.bucket_name, Key=output_s3_folder)
        print(f"Created output folder in S3: s3://{s3_manager.bucket_name}/{output_s3_folder}")
    except Exception as e:
        print(f"Error creating output folder in S3: {e}")
        raise

    return f"s3://{s3_manager.bucket_name}/{input_s3_key}", output_s3_folder


def upload_vocabulary_files(transcribe_client, vocab_name: str, vocab_file: str):
    """
    Uploads a custom vocabulary file to AWS Transcribe.

    Parameters:
    - transcribe_client: The AWS Transcribe client.
    - vocab_name (str): The name of the vocabulary to create or update.
    - vocab_file (str): The local file path of the vocabulary CSV.

    Returns:
    - None
    """
    try:
        # Check if the vocabulary already exists
        response = transcribe_client.get_vocabulary(VocabularyName=vocab_name)
        if response['VocabularyState'] in ['READY', 'PENDING']:
            print(f"Vocabulary '{vocab_name}' already exists and is ready or pending. Skipping upload.")
            return
    except transcribe_client.exceptions.BadRequestException:
        print(f"Vocabulary '{vocab_name}' does not exist. Proceeding to create it.")
        raise("Error: I need to setup uploading file in .txt form to s3 first - Phrases does not work. Use VocabularyFileUri instead")
    
    # Upload vocabulary
    with open(vocab_file, 'r') as file:
        vocab_content = file.read()

    try:
        transcribe_client.create_vocabulary(
            VocabularyName=vocab_name,
            LanguageCode=LANGUAGE_CODE,
            Phrases=vocab_content.splitlines(),
        )
        print(f"Vocabulary '{vocab_name}' has been uploaded successfully.")
    except Exception as e:
        print(f"Error uploading vocabulary '{vocab_name}': {e}")
        raise


def start_aws_transcription_job(transcription_manager, input_s3_uri: str, output_s3_dir_prefix: str) -> str:
    """Starts a transcription job and returns the job name."""
    transcribe_client = transcription_manager.transcribe_client
    # Upload custom vocabularies
    upload_vocabulary_files(transcribe_client, "FP-SA-Vocab-V1", TRANSCRIBE_VOCABULARY_FILE)

    job_name = f"transcription_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": input_s3_uri},
            MediaFormat=MEDIA_FORMAT,
            LanguageCode=LANGUAGE_CODE,
            OutputBucketName=DEFAULT_AWS_BUCKET_NAME,
            OutputKey=output_s3_dir_prefix,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": DEFAULT_TRANSCRIBE_MAX_SPEAKERS,
                "ShowAlternatives": False,
                "VocabularyName": "FP-SA-Vocab-V1",
            },
        )
        print(f"Transcription job '{job_name}' started successfully.")
        return job_name
    except Exception as e:
        print(f"Error starting transcription job '{job_name}': {e}")
        raise

