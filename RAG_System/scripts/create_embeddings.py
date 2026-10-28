import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

# =====================================================
# LOAD DOCUMENTS
# =====================================================

ROOT_FOLDER = r"C:\Users\User\OneDrive\Desktop\FYP\RAG_System"

SOP_FOLDER = os.path.join(ROOT_FOLDER, "data", "SOP")
RCA_FOLDER = os.path.join(ROOT_FOLDER, "data", "RCA")

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

# =====================================================
# CHUNK
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print("Chunks:", len(chunks))

# =====================================================
# EMBEDDING MODEL
# =====================================================

print("Loading Embedding Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model Loaded")

# =====================================================
# CREATE EMBEDDINGS
# =====================================================

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embeddings Generated")

print("Number of Embeddings:",
      len(embeddings))

print("Embedding Dimension:",
      len(embeddings[0]))

# =====================================================
# PREVIEW
# =====================================================

print("\nFirst Chunk Source:")
print(chunks[0].metadata["source"])

print("\nFirst Embedding Length:")
print(len(embeddings[0]))
