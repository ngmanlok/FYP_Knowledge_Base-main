import os
from langchain_community.document_loaders import Docx2txtLoader

# =====================================================
# ABSOLUTE PATH VERSION
# =====================================================

ROOT_FOLDER = r"C:\Users\User\OneDrive\Desktop\FYP\RAG_System"

SOP_FOLDER = os.path.join(ROOT_FOLDER, "data", "SOP")
RCA_FOLDER = os.path.join(ROOT_FOLDER, "data", "RCA")

print("=" * 60)
print("PATH CHECK")
print("=" * 60)

print("SOP Folder:", SOP_FOLDER)
print("SOP Exists:", os.path.exists(SOP_FOLDER))

print("RCA Folder:", RCA_FOLDER)
print("RCA Exists:", os.path.exists(RCA_FOLDER))

all_docs = []

# =====================================================
# SOP
# =====================================================

if os.path.exists(SOP_FOLDER):

    print("\nLoading SOP Documents...")

    for filename in os.listdir(SOP_FOLDER):

        if filename.endswith(".docx"):

            try:

                file_path = os.path.join(SOP_FOLDER, filename)

                loader = Docx2txtLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["document_type"] = "SOP"

                all_docs.extend(docs)

                print(f"[SUCCESS] {filename}")

            except Exception as e:

                print(f"[FAILED] {filename}")
                print(e)

# =====================================================
# RCA
# =====================================================

if os.path.exists(RCA_FOLDER):

    print("\nLoading RCA Documents...")

    for filename in os.listdir(RCA_FOLDER):

        if filename.endswith(".docx"):

            try:

                file_path = os.path.join(RCA_FOLDER, filename)

                loader = Docx2txtLoader(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["document_type"] = "RCA"

                all_docs.extend(docs)

                print(f"[SUCCESS] {filename}")

            except Exception as e:

                print(f"[FAILED] {filename}")
                print(e)

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 60)
print("LOADING COMPLETE")
print("=" * 60)

print("Documents Loaded:", len(all_docs))

if len(all_docs) > 0:

    print("\nFirst Document:")
    print(all_docs[0].metadata)
    print("\nPreview:")
    print(all_docs[0].page_content[:300])

else:

    print("No documents loaded.")
