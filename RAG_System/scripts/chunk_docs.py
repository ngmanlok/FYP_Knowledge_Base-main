import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# PATHS
# =====================================================

ROOT_FOLDER = r"C:\Users\User\OneDrive\Desktop\FYP\RAG_System"

SOP_FOLDER = os.path.join(ROOT_FOLDER, "data", "SOP")
RCA_FOLDER = os.path.join(ROOT_FOLDER, "data", "RCA")

# =====================================================
# LOAD DOCUMENTS
# =====================================================

all_docs = []

for folder, doc_type in [
    (SOP_FOLDER, "SOP"),
    (RCA_FOLDER, "RCA")
]:

    for filename in os.listdir(folder):

        if filename.endswith(".docx"):

            file_path = os.path.join(folder, filename)

            loader = Docx2txtLoader(file_path)

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["document_type"] = doc_type

            all_docs.extend(docs)

print("=" * 60)
print("DOCUMENTS LOADED")
print("=" * 60)
print("Total Documents:", len(all_docs))

# =====================================================
# CHUNK DOCUMENTS
# =====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print("\n")
print("=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print("Total Chunks:", len(chunks))

# =====================================================
# SHOW FIRST 5 CHUNKS
# =====================================================

for i in range(min(5, len(chunks))):

    print("\n")
    print("-" * 60)

    print("Chunk:", i + 1)
    print("Source:", chunks[i].metadata["source"])
    print("Type:", chunks[i].metadata["document_type"])

    print("\nContent:\n")

    print(chunks[i].page_content[:300])

# =====================================================
# SAVE CHUNKS
# =====================================================

os.makedirs("../vector_db", exist_ok=True)

with open("../vector_db/chunks_preview.txt",
          "w",
          encoding="utf-8") as f:

    for i, chunk in enumerate(chunks):

        f.write(f"\n===== CHUNK {i+1} =====\n")

        f.write(
            f"Source: {chunk.metadata['source']}\n"
        )

        f.write(
            f"Type: {chunk.metadata['document_type']}\n\n"
        )

        f.write(chunk.page_content)
        f.write("\n")

print("\n")
print("chunks_preview.txt created successfully")