import numpy as np

from ingest import load_markdown_file
from chunk import create_chunks
from embed import create_embeddings
from vector_store import build_index


MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve(query: str, chunks: list[dict], index, model, top_k: int = 3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        np.asarray(query_embedding, dtype="float32"),
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        results.append({
            "chunk": chunks[index_position],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":

    from sentence_transformers import SentenceTransformer

    document = load_markdown_file(
        "data/requirements/login.md"
    )

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    query = "What validation happens if the user doesn't provide an email?"

    results = retrieve(
        query=query,
        chunks=chunks,
        index=index,
        model=model,
        top_k=3
    )

    print("\nQUERY:")
    print(query)

    print("\nRETRIEVED CONTEXT:")

    for i, result in enumerate(results, start=1):

        print("\n==============================")
        print(f"RESULT {i}")
        print("==============================")

        print("Distance:", result["distance"])
        print("Chunk ID:", result["chunk"]["chunk_id"])
        print("\n", result["chunk"]["text"])