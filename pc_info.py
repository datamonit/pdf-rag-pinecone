import os
from dotenv import load_dotenv
from pinecone_utils import list_indexes, describe_index, delete_index

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag")
## delete_index(INDEX_NAME)

print(list_indexes())

print(f"Describing Pinecone index: {INDEX_NAME}")
details = describe_index(INDEX_NAME)
print(details)
