import os
from pinecone import Pinecone, ServerlessSpec
from llm_utils import embed_text

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)


def init_index(index_name: str, dimension: int = 1536):
    """Create Pinecone index if it does not exist"""
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def list_indexes():
    """Return a list of all existing Pinecone indexes."""
    return [idx["name"] for idx in pc.list_indexes()]


def describe_index(index_name: str):
    """Return metadata/details of a specific Pinecone index."""
    try:
        return pc.describe_index(index_name)
    except Exception as e:
        print(f"Error describing index '{index_name}': {e}")
        return None


def delete_index(index_name: str):
    """Delete a Pinecone index by name."""
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"Deleting index: {index_name}")
        pc.delete_index(index_name)
    else:
        print(f"Index '{index_name}' does not exist, nothing to delete.")

    print(f"✅ Deleted index: {index_name}")


def upsert_chunks(index, chunks, namespace: str, pdf_path: str):
    """Insert PDF chunks into Pinecone index under a namespace"""
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk.page_content)
        metadata = {"filename": pdf_path, "page": chunk.metadata.get("page", 0)}
        index.upsert(
            vectors=[
                {"id": f"{namespace}-chunk-{i}", "values": vector, "metadata": metadata}
            ],
            namespace=namespace,
        )
    print(f"✅ Inserted {len(chunks)} chunks into namespace '{namespace}'")


def query_index(index, question: str, namespace: str, top_k: int = 3):
    """Query Pinecone index with namespace restriction"""
    q_vector = embed_text(question)

    results = index.query(
        vector=q_vector, top_k=top_k, include_metadata=True, namespace=namespace
    )
    return results
