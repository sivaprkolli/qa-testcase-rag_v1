from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import load_markdown_file


def create_chunks(document: dict) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(document["text"])

    results = []

    for index, chunk in enumerate(chunks):

        results.append({
            "chunk_id": f"{document['source']}-{index}",
            "text": chunk,
            "source": document["source"],
            "document_type": document["document_type"]
        })

    return results


if __name__ == "__main__":

    document = load_markdown_file(
        "data/requirements/login.md"
    )

    chunks = create_chunks(document)

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print("\n--------------------")
        print("Chunk ID:", chunk["chunk_id"])
        print(chunk["text"])