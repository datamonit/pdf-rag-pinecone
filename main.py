from pdf_utils import load_and_split_pdf
from pinecone_utils import init_index, delete_index, upsert_chunks, query_index
from llm_utils import generate_answer

# Config
INDEX_NAME = "pdf-rag-index"
PDF_PATH = "MC0101072019.pdf"
NAMESPACE = "finance-pdfs"

# Load and Split PDF
chunks = load_and_split_pdf(PDF_PATH)

# Setup Pinecone Index
index = init_index(INDEX_NAME)

# Upsert Chunks
upsert_chunks(index, chunks, namespace=NAMESPACE, pdf_path=PDF_PATH)

# Query RAG
question = "What does RBI recommend for Preservation of Counterfeit Notes Received from Police Authorities?"
results = query_index(index, question, namespace=NAMESPACE)

# Build context from results
retrieved_sources = []
context_texts = []

for match in results["matches"]:
    retrieved_sources.append(
        f"Page {match['metadata'].get('page')} from {match['metadata'].get('filename')} (score: {match['score']:.2f})"
    )
    context_texts.append(
        f"[p{match['metadata'].get('page')}] " + str(match["metadata"])
    )

context_str = "\n".join(context_texts)

# Generate final answer
answer = generate_answer(question, context_str, retrieved_sources)

print("\n❓ Question:", question)
print("💡 Answer:", answer)

# Cleanup
# delete_index(INDEX_NAME)
delete_confirmation = input(
    f"\nDo you really want to DELETE the index '{INDEX_NAME}'? Type YES to confirm: "
)
if delete_confirmation.strip().upper() == "YES":
    delete_index(INDEX_NAME)
    print(f"Index '{INDEX_NAME}' deleted.")
else:
    print("Deletion cancelled.")

print("\n✅ Done!")
