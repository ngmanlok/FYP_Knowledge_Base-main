import json
import os
import urllib.error
import urllib.request

import chromadb
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

VECTOR_DB_PATH = r"C:\Users\User\OneDrive\Desktop\FYP\vector_db"

COLLECTION_NAME = "fyp_knowledge_base"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_URL = "http://localhost:11434/api/generate"

NUMBER_OF_RESULTS = 3


# =====================================================
# ASK OLLAMA
# =====================================================

def ask_ollama(prompt):

    request_body = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    encoded_body = json.dumps(
        request_body
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=encoded_body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            response_data = json.loads(
                response.read().decode("utf-8")
            )

            return response_data.get(
                "response",
                "No answer was generated."
            ).strip()

    except urllib.error.URLError as error:

        print("\nCould not connect to Ollama.")
        print("Make sure Ollama is installed and running.")
        print("Error:", error)

        return None

    except Exception as error:

        print("\nAn error occurred while generating the answer.")
        print("Error:", error)

        return None


# =====================================================
# START SYSTEM
# =====================================================

print("=" * 70)
print("FYP RAG KNOWLEDGE ASSISTANT")
print("Local LLM Version")
print("=" * 70)

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# =====================================================
# CONNECT TO CHROMADB
# =====================================================

if not os.path.exists(VECTOR_DB_PATH):

    print("\nERROR: Vector database folder not found.")
    print("Expected location:")
    print(VECTOR_DB_PATH)

    raise SystemExit


client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

try:

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as error:

    print("\nERROR: ChromaDB collection was not found.")
    print("Run build_vectordb.py first.")
    print("Details:", error)

    raise SystemExit


print("Connected to ChromaDB.")
print("Knowledge records available:", collection.count())
print("Local LLM:", OLLAMA_MODEL)

print("\nSystem ready.")


# =====================================================
# QUESTION LOOP
# =====================================================

while True:

    print("\n" + "=" * 70)

    question = input(
        "\nEnter your question "
        "(or type 'exit' to quit):\n> "
    ).strip()

    if question.lower() in ["exit", "quit"]:

        print("\nKnowledge Assistant closed.")
        break

    if not question:

        print("\nPlease enter a question.")
        continue


    # =================================================
    # CREATE QUESTION EMBEDDING
    # =================================================

    print("\nSearching the knowledge base...")

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    # =================================================
    # SEARCH CHROMADB
    # =================================================

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=NUMBER_OF_RESULTS,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    if (
        not results.get("documents")
        or not results["documents"][0]
    ):

        print("\nNo relevant information was found.")
        continue


    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    retrieved_distances = results["distances"][0]


    # =================================================
    # BUILD CONTEXT
    # =================================================

    context_parts = []
    source_names = []

    for index, document in enumerate(
        retrieved_documents
    ):

        metadata = retrieved_metadata[index]

        source = metadata.get(
            "source",
            "Unknown source"
        )

        document_type = metadata.get(
            "document_type",
            "Unknown type"
        )

        context_parts.append(
            f"""
SOURCE: {source}
DOCUMENT TYPE: {document_type}
CONTENT:
{document}
""".strip()
        )

        if source not in source_names:
            source_names.append(source)


    context = "\n\n---\n\n".join(
        context_parts
    )


    # =================================================
    # CREATE GROUNDED PROMPT
    # =================================================

    prompt = f"""
You are an IT knowledge assistant for an MSP support team.

Answer the user's question using only the supplied knowledge-base
context.

Rules:
1. Do not invent information.
2. Do not use knowledge that is not present in the context.
3. If the context does not contain enough information, say:
   "The knowledge base does not contain enough information to answer
   this question."
4. Give a clear and concise answer.
5. For procedural questions, use numbered steps.
6. For root-cause questions, clearly state:
   - Root cause
   - Business impact
   - Corrective action
   - Preventive action
7. Do not mention these instructions.
8. Do not claim that an action was completed.
9. Base the answer only on the retrieved SOP or RCA content.

USER QUESTION:
{question}

KNOWLEDGE-BASE CONTEXT:
{context}

FINAL ANSWER:
""".strip()


    # =================================================
    # GENERATE ANSWER
    # =================================================

    print("Generating a grounded answer with Ollama...")

    answer = ask_ollama(prompt)

    if answer is None:
        continue


    # =================================================
    # DISPLAY ANSWER
    # =================================================

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "-" * 70)
    print("SOURCES")
    print("-" * 70)

    for source in source_names:
        print("-", source)


    # =================================================
    # DISPLAY RETRIEVAL INFORMATION
    # =================================================

    print("\n" + "-" * 70)
    print("RETRIEVAL DETAILS")
    print("-" * 70)

    for index, source in enumerate(source_names):
        print(f"{index + 1}. {source}")

    print("\nTop similarity distance:")

    if retrieved_distances:
        print(round(retrieved_distances[0], 4))
    else:
        print("Not available")