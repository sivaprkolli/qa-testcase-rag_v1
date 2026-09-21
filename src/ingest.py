from pathlib import Path


def load_markdown_file(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "source": path.name,
        "file_path": str(path),
        "document_type": "requirement"
    }


if __name__ == "__main__":
    document = load_markdown_file(
        "data/requirements/login.md"
    )

    print("Source:", document["source"])
    print("Document Type:", document["document_type"])
    print("\nContent:\n")
    print(document["text"])