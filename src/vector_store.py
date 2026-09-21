import faiss
import numpy as np

from ingest import load_markdown_file
from chunk import create_chunks
from embed import create_embeddings


def build_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return index


if __name__ == "__main__":

    document = load_markdown_file(
        "data/requirements/login.md"
    )

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    print("Vector dimension:", index.d)
    print("Number of vectors:", index.ntotal)