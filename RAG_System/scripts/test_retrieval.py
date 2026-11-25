import os
import chromadb
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

ROOT_FOLDER = r"C:\Users\User\OneDrive\Desktop\FYP\RAG_System"

VECTOR_DB_PATH = r"C:\Users\User\OneDrive\Desktop\FYP\vector_db"

COLLECTION_NAME = "fyp_knowledge_base"


# =====================================================
# START
# =====================================================

print("=" * 70)
print("FYP RAG KNOWLEDGE ASSISTANT")
print("=" * 70)

print("\nLoading embedding model...")


# =====================================================
# LOAD EMBEDDING MODEL
# Must be the SAME model used when building ChromaDB
# =====================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =====================================================
# CONNECT TO CHROMADB
# =====================================================

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Connected to ChromaDB.")
print("Knowledge records available:", collection.count())


# =====================================================
# QUESTION LOOP
# =====================================================

while True:

    print("\n" + "=" * 70)

    question = input(
        "\nEnter your question (or type 'exit' to quit):\n> "
    ).strip()

    if question.lower() == "exit":
        print("\nKnowledge Assistant closed.")
        break

    if not question:
        print("\nPlease enter a question.")
        continue


    # =================================================
    # CONVERT QUESTION INTO EMBEDDING
    # =================================================

    query_embedding = model.encode(
        question
    ).tolist()


    # =================================================
    # SEARCH CHROMADB
    # =================================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )


    # =================================================
    # DISPLAY BEST RESULT
    # =================================================

    print("\nSearching knowledge base...")

    if not results["documents"] or not results["documents"][0]:

        print("\nNo relevant information found.")
        continue


    best_document = results["documents"][0][0]
    best_metadata = results["metadatas"][0][0]

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(best_document)

    print("\n" + "-" * 70)
    print("SOURCE")
    print("-" * 70)

    print(best_metadata.get(
        "source",
        "Unknown"
    ))

    print("\nDOCUMENT TYPE")
    print("-" * 70)

    print(best_metadata.get(
        "document_type",
        "Unknown"
    ))


    # =================================================
    # SHOW OTHER RETRIEVED SOURCES
    # =================================================

    print("\nOTHER RELEVANT RESULTS")
    print("-" * 70)

    for i in range(
        1,
        len(results["documents"][0])
    ):

        metadata = results["metadatas"][0][i]

        print(
            f"{i + 1}. "
            f"{metadata.get('source', 'Unknown')}"
        )