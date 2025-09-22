import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Setup OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str):
    """Generate embeddings using OpenAI embedding model"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    )
    return response.data[0].embedding


def generate_answer(question: str, context: str, retrieved_sources: list[str]):
    """Generate final answer from GPT model"""
    prompt = f"""Answer the following question using the context below:

    Question: {question}

    Context:
    {context}

    Retrieved Sources:
    {retrieved_sources}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content
