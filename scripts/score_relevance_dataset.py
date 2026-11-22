
import csv
from pathlib import Path
from statistics import mean

from sentence_transformers import CrossEncoder


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_relevance_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "rag_relevance_scored.csv"
)

# Use the same model as your existing RAG application
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    # 1. Load dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Dataset is empty.")

    # 2. Prepare question-paragraph pairs
    sentence_pairs = [
        (row["question"], row["paragraph"])
        for row in rows
    ]

    # 3. Load Cross-Encoder
    print("Loading Cross-Encoder model...")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu"
    )

    # 4. Calculate relevance scores
    print(f"Scoring {len(sentence_pairs)} pairs...")

    scores = model.predict(
        sentence_pairs,
        batch_size=8,
        show_progress_bar=True
    )

    # 5. Add scores to dataset
    for row, score in zip(rows, scores):
        row["cross_encoder_score"] = float(score)

    # 6. Save scored dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 7. Display score statistics
    relevant_scores = [
        row["cross_encoder_score"]
        for row in rows
        if row["label"].strip().lower() == "relevant"
    ]

    irrelevant_scores = [
        row["cross_encoder_score"]
        for row in rows
        if row["label"].strip().lower() == "irrelevant"
    ]

    print("\n========== SCORE SUMMARY ==========")

    print(f"Total pairs: {len(rows)}")

    if relevant_scores:
        print("\nRelevant pairs:")
        print(f"Count: {len(relevant_scores)}")
        print(f"Minimum: {min(relevant_scores):.4f}")
        print(f"Maximum: {max(relevant_scores):.4f}")
        print(f"Average: {mean(relevant_scores):.4f}")

    if irrelevant_scores:
        print("\nIrrelevant pairs:")
        print(f"Count: {len(irrelevant_scores)}")
        print(f"Minimum: {min(irrelevant_scores):.4f}")
        print(f"Maximum: {max(irrelevant_scores):.4f}")
        print(f"Average: {mean(irrelevant_scores):.4f}")

    print("\nScored dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
