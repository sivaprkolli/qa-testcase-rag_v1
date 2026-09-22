import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Allow imports from src/
sys.path.append(
    str(Path(__file__).resolve().parents[1] / "src")
)

from ingest import load_markdown_file
from chunk import create_chunks
from embed import create_embeddings
from vector_store import build_index
from retrieve import retrieve


MODEL_NAME = "all-MiniLM-L6-v2"


EVALUATION_DATASET = [
    {
        "query": "What happens when the user enters an invalid password?",
        "expected_ac_ids": {"AC-02"},
    },
    {
        "query": "What happens when an unregistered email is used?",
        "expected_ac_ids": {"AC-03"},
    },
    {
        "query": "What happens when the email field is empty?",
        "expected_ac_ids": {"AC-04"},
    },
    {
        "query": "What happens when the password field is empty?",
        "expected_ac_ids": {"AC-05"},
    },
    {
        "query": "How does a successful login work?",
        "expected_ac_ids": {"AC-01"},
    },
]


def extract_ac_ids(text: str) -> set[str]:
    """
    Extract AC identifiers from chunk text.
    Example: 'AC-02 Invalid Password' -> {'AC-02'}
    """

    return set(
        re.findall(r"\bAC-\d+\b", text)
    )


def build_retrieval_pipeline():

    document = load_markdown_file(
        "data/requirements/login.md"
    )

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    return chunks, index, model


def evaluate_query(
    query: str,
    expected_ac_ids: set[str],
    chunks: list[dict],
    index,
    model,
    top_k: int = 3,
):

    results = retrieve(
        query=query,
        chunks=chunks,
        index=index,
        model=model,
        top_k=top_k,
    )

    actual_ac_ids = set()

    retrieved_chunk_ids = []

    for result in results:

        chunk = result["chunk"]

        retrieved_chunk_ids.append(
            chunk["chunk_id"]
        )

        actual_ac_ids.update(
            extract_ac_ids(chunk["text"])
        )

    true_positives = (
        expected_ac_ids & actual_ac_ids
    )

    false_positives = (
        actual_ac_ids - expected_ac_ids
    )

    false_negatives = (
        expected_ac_ids - actual_ac_ids
    )

    precision = (
        len(true_positives)
        / len(actual_ac_ids)
        if actual_ac_ids
        else 0.0
    )

    recall = (
        len(true_positives)
        / len(expected_ac_ids)
        if expected_ac_ids
        else 0.0
    )

    return {
        "query": query,
        "expected": expected_ac_ids,
        "actual": actual_ac_ids,
        "retrieved_chunks": retrieved_chunk_ids,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
    }


def main():

    chunks, index, model = build_retrieval_pipeline()

    results = []

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    for item in EVALUATION_DATASET:

        result = evaluate_query(
            query=item["query"],
            expected_ac_ids=item["expected_ac_ids"],
            chunks=chunks,
            index=index,
            model=model,
            top_k=3,
        )

        results.append(result)

        print("Query:")
        print(result["query"])

        print("\nExpected ACs:")
        print(result["expected"])

        print("\nActual ACs:")
        print(result["actual"])

        print("\nRetrieved chunks:")
        print(result["retrieved_chunks"])

        print("\nTrue positives:")
        print(result["true_positives"])

        print("False positives:")
        print(result["false_positives"])

        print("False negatives:")
        print(result["false_negatives"])

        print(
            f"\nPrecision: {result['precision']:.2f}"
        )

        print(
            f"Recall:    {result['recall']:.2f}"
        )

        print("\n" + "-" * 60)

    evaluate_overall(results)


def evaluate_overall(results):

    total_precision = np.mean(
        [result["precision"] for result in results]
    )

    total_recall = np.mean(
        [result["recall"] for result in results]
    )

    print("\n========== OVERALL RESULTS ==========\n")

    print(
        f"Average Precision: {total_precision:.2f}"
    )

    print(
        f"Average Recall:    {total_recall:.2f}"
    )


if __name__ == "__main__":
    main()