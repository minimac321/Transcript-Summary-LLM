import pinecone
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY')
)

# List all indexes
index_list = pinecone.list_indexes()

# Get details of a specific index
if index_list:
    index_name = index_list[0]  # Get the first index
    index_info = pinecone.describe_index(index_name)
    print(index_info)
else:
    print("No indexes found.")