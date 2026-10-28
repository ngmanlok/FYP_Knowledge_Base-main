import os
from langchain_community.document_loaders import Docx2txtLoader

# -----------------------------
# Folder locations
# -----------------------------
SOP_FOLDER = "../data/SOP"
RCA_FOLDER = "../data/RCA"

all_docs = []

print("=" * 60)
print("Starting Document Loading")
print("=" * 60)

# -----------------------------
# SOP Documents
# -----------------------------
if os.path.exists(SOP_FOLDER):
    print("\nLoading SOP Documents...")

    for filename in os.listdir(SOP_FOLDER):

        if filename.endswith(".docx"):

            file_path = os.path.join(SOP_FOLDER, filename)

            try:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["document_type"] = "SOP"

                all_docs.extend(docs)

                print(f"[SUCCESS] {filename}")

            except Exception as e:
                print(f"[FAILED] {filename}")
                print(e)


# -----------------------------
# RCA Documents
# -----------------------------
if os.path.exists(RCA_FOLDER):
    print("\nLoading RCA Documents...")

    for filename in os.listdir(RCA_FOLDER):

        if filename.endswith(".docx"):

            file_path = os.path.join(RCA_FOLDER, filename)

            try:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["document_type"] = "RCA"

                all_docs.extend(docs)

                print(f"[SUCCESS] {filename}")

            except Exception as e:
                print(f"[FAILED] {filename}")
                print(e)


# -----------------------------
# Summary
# -----------------------------
print("\n")
print("=" * 60)
print("LOADING COMPLETE")
print("=" * 60)

print(f"Documents Loaded: {len(all_docs)}")


# -----------------------------
# Preview First Document
# -----------------------------
if len(all_docs) > 0:

    first_doc = all_docs[0]

    print("\nFIRST DOCUMENT PREVIEW")
    print("-" * 60)

    print("Source:")
    print(first_doc.metadata["source"])

    print("\nDocument Type:")
    print(first_doc.metadata["document_type"])

    print("\nFirst 500 Characters:")
    print("-" * 60)

    print(first_doc.page_content[:500])

else:
    print("No documents loaded.")