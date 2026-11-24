import os
import chromadb

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ====================================
# PATHS
# ====================================

ROOT_FOLDER = r"C:\Users\User\OneDrive\Desktop\FYP\RAG_System"

SOP_FOLDER = os.path.join(ROOT_FOLDER, "data", "SOP")
RCA_FOLDER = os.path.join(ROOT_FOLDER, "data", "RCA")

# ====================================
# LOAD DOCUMENTS
# ====================================

all_docs = []

for folder, doc_type in [
    (SOP_FOLDER, "SOP"),
    (RCA_FOLDER, "RCA")
]:

    for filename in os.listdir(folder):

        if filename.endswith(".docx"):

            loader = Docx2txtLoader(
                os.path.join(folder, filename)
            )

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["document_type"] = doc_type

            all_docs.extend(docs)

print("Documents:", len(all_docs))

# ====================================
# CHUNKING
# ====================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print("Chunks:", len(chunks))

# ====================================
# EMBEDDINGS
# ====================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embeddings:", len(embeddings))

# ====================================
# CHROMADB
# ====================================

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    name="fyp_knowledge_base"
)

# ====================================
# STORE VECTORS
# ====================================

for i, chunk in enumerate(chunks):

    collection.add(
        ids=[f"chunk_{i}"],

        embeddings=[
            embeddings[i].tolist()
        ],

        documents=[
            chunk.page_content
        ],

        metadatas=[{
            "source":
                chunk.metadata["source"],

            "document_type":
                chunk.metadata["document_type"]
        }]
    )

print("\n✅ ChromaDB Created")

print(
    "Total Records:",
    collection.count()
)