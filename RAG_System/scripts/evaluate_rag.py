import os
import re
import sys
from collections import defaultdict

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = r"C:\Users\User\OneDrive\Desktop\FYP"
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")
EVALUATION_FILE = os.path.join(
    PROJECT_ROOT,
    "Evaluation_Question",
    "Evaluation_Questions.xlsx"
)

OUTPUT_FILE = os.path.join(PROJECT_ROOT, "RAG_Evaluation_Results.xlsx")

COLLECTION_NAME = "fyp_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


# =====================================================
# HELPERS
# =====================================================

def normalize_document_id(value):
    """Extract IDs such as SOP001 or RCA001 from filenames or expected values."""

    if value is None:
        return "UNKNOWN"

    text = str(value).upper()

    match = re.search(
        r"(SOP|RCA)\s*0*(\d{1,3})(?!\d)",
        text
    )

    if not match:
        return "UNKNOWN"

    prefix = match.group(1)
    number = int(match.group(2))

    return f"{prefix}{number:03d}"


def require_columns(dataframe, required_columns):
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Evaluation file is missing required columns: " + ", ".join(missing)
        )


def safe_percentage(correct, total):
    return round((correct / total) * 100, 2) if total else 0.0


# =====================================================
# VALIDATE PATHS
# =====================================================

print("=" * 72)
print("FYP RAG RETRIEVAL EVALUATION")
print("=" * 72)

if not os.path.isdir(VECTOR_DB_PATH):
    print("\nERROR: ChromaDB folder was not found:")
    print(VECTOR_DB_PATH)
    sys.exit(1)

if not os.path.isfile(EVALUATION_FILE):
    print("\nERROR: Evaluation workbook was not found:")
    print(EVALUATION_FILE)
    print("\nPlace Evaluation_Questions.xlsx in the FYP root folder or update EVALUATION_FILE.")
    sys.exit(1)


# =====================================================
# LOAD EVALUATION QUESTIONS
# =====================================================

print("\nLoading evaluation questions...")

evaluation_df = pd.read_excel(
    EVALUATION_FILE,
    engine="openpyxl"
)

evaluation_df.columns = [str(column).strip() for column in evaluation_df.columns]

required_columns = [
    "ID",
    "Question",
    "Expected Document",
    "Type",
    "Difficulty",
]
require_columns(evaluation_df, required_columns)

evaluation_df = evaluation_df.dropna(subset=["Question", "Expected Document"]).copy()
evaluation_df["Question"] = evaluation_df["Question"].astype(str).str.strip()
evaluation_df["Expected Document"] = (
    evaluation_df["Expected Document"].astype(str).str.strip()
)

print(f"Questions loaded: {len(evaluation_df)}")


# =====================================================
# LOAD EMBEDDING MODEL AND CHROMADB
# =====================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model loaded.")

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as error:
    print("\nERROR: ChromaDB collection was not found.")
    print("Run build_vectordb.py first.")
    print("Details:", error)
    sys.exit(1)

print(f"Connected to ChromaDB: {collection.count()} records available")
print(f"Evaluation mode: Top-{TOP_K}")


# =====================================================
# RUN RETRIEVAL EVALUATION
# =====================================================

results_rows = []
summary_by_difficulty = defaultdict(lambda: {"total": 0, "top1": 0, "topk": 0})
summary_by_type = defaultdict(lambda: {"total": 0, "top1": 0, "topk": 0})

top1_correct_total = 0
topk_correct_total = 0

for position, row in evaluation_df.reset_index(drop=True).iterrows():
    question_id = row["ID"]
    question = row["Question"]
    expected_raw = row["Expected Document"]
    question_type = str(row["Type"]).strip()
    difficulty = str(row["Difficulty"]).strip()
    expected_id = normalize_document_id(expected_raw)

    query_embedding = embedding_model.encode(question).tolist()

    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = query_result.get("metadatas", [[]])[0] or []
    distances = query_result.get("distances", [[]])[0] or []

    retrieved_sources = []
    retrieved_ids = []

    for metadata in metadatas:
        metadata = metadata or {}
        source = str(metadata.get("source", "Unknown"))
        retrieved_sources.append(source)
        retrieved_ids.append(normalize_document_id(source))

    while len(retrieved_sources) < TOP_K:
        retrieved_sources.append("")
        retrieved_ids.append("UNKNOWN")

    while len(distances) < TOP_K:
        distances.append(None)

    top1_correct = bool(retrieved_ids and retrieved_ids[0] == expected_id)
    topk_correct = expected_id in retrieved_ids[:TOP_K]

    top1_correct_total += int(top1_correct)
    topk_correct_total += int(topk_correct)

    summary_by_difficulty[difficulty]["total"] += 1
    summary_by_difficulty[difficulty]["top1"] += int(top1_correct)
    summary_by_difficulty[difficulty]["topk"] += int(topk_correct)

    summary_by_type[question_type]["total"] += 1
    summary_by_type[question_type]["top1"] += int(top1_correct)
    summary_by_type[question_type]["topk"] += int(topk_correct)

    results_rows.append({
        "ID": question_id,
        "Question": question,
        "Expected Document": expected_id,
        "Type": question_type,
        "Difficulty": difficulty,
        "Top 1 Retrieved": retrieved_ids[0],
        "Top 1 Source": retrieved_sources[0],
        "Top 1 Distance": distances[0],
        "Top 1 Result": "PASS" if top1_correct else "FAIL",
        "Top 2 Retrieved": retrieved_ids[1],
        "Top 2 Source": retrieved_sources[1],
        "Top 2 Distance": distances[1],
        "Top 3 Retrieved": retrieved_ids[2],
        "Top 3 Source": retrieved_sources[2],
        "Top 3 Distance": distances[2],
        f"Top {TOP_K} Result": "PASS" if topk_correct else "FAIL",
    })

    status = "PASS" if top1_correct else "FAIL"
    print(
        f"[{position + 1:02d}/{len(evaluation_df):02d}] "
        f"{status} | Expected: {expected_id} | Retrieved: {retrieved_ids[0]}"
    )


# =====================================================
# BUILD SUMMARY TABLES
# =====================================================

results_df = pd.DataFrame(results_rows)
total_questions = len(results_df)

overall_summary_df = pd.DataFrame([
    {
        "Metric": "Total Questions",
        "Value": total_questions,
    },
    {
        "Metric": "Top 1 Correct",
        "Value": top1_correct_total,
    },
    {
        "Metric": "Top 1 Incorrect",
        "Value": total_questions - top1_correct_total,
    },
    {
        "Metric": "Top 1 Accuracy (%)",
        "Value": safe_percentage(top1_correct_total, total_questions),
    },
    {
        "Metric": f"Top {TOP_K} Correct",
        "Value": topk_correct_total,
    },
    {
        "Metric": f"Top {TOP_K} Accuracy (%)",
        "Value": safe_percentage(topk_correct_total, total_questions),
    },
])


def make_breakdown(summary_dictionary, category_name):
    rows = []
    for category in sorted(summary_dictionary):
        values = summary_dictionary[category]
        rows.append({
            category_name: category,
            "Total": values["total"],
            "Top 1 Correct": values["top1"],
            "Top 1 Accuracy (%)": safe_percentage(values["top1"], values["total"]),
            f"Top {TOP_K} Correct": values["topk"],
            f"Top {TOP_K} Accuracy (%)": safe_percentage(values["topk"], values["total"]),
        })
    return pd.DataFrame(rows)


difficulty_df = make_breakdown(summary_by_difficulty, "Difficulty")
type_df = make_breakdown(summary_by_type, "Question Type")


# =====================================================
# SAVE EXCEL REPORT
# =====================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    overall_summary_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
    difficulty_df.to_excel(writer, sheet_name="Summary", index=False, startrow=9)
    type_df.to_excel(writer, sheet_name="Summary", index=False, startrow=16)
    results_df.to_excel(writer, sheet_name="Detailed Results", index=False)

    workbook = writer.book

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for column_cells in worksheet.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_length = max(max_length, len(value))

            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 55)

    summary_sheet = workbook["Summary"]
    summary_sheet["A9"] = "Accuracy by Difficulty"
    summary_sheet["A16"] = "Accuracy by Question Type"


# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n" + "=" * 72)
print("EVALUATION COMPLETE")
print("=" * 72)
print(f"Total questions: {total_questions}")
print(f"Top 1 correct: {top1_correct_total}")
print(f"Top 1 accuracy: {safe_percentage(top1_correct_total, total_questions)}%")
print(f"Top {TOP_K} correct: {topk_correct_total}")
print(f"Top {TOP_K} accuracy: {safe_percentage(topk_correct_total, total_questions)}%")
print("\nReport saved to:")
print(OUTPUT_FILE)
