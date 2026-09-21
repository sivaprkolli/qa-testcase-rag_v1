from sentence_transformers import SentenceTransformer

from ingest import load_markdown_file
from chunk import create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks: list[dict]):
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":

    document = load_markdown_file(
        "data/requirements/login.md"
    )

    chunks = create_chunks(document)

    embeddings = create_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("First embedding:")
    print(embeddings[0])

    print("Second embedding:")
    print(embeddings[1])