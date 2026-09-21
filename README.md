# qa-testcase-rag

A Retrieval-Augmented Generation (RAG) pipeline that generates software QA test cases from requirement documents. It retrieves the most relevant chunks of a requirement spec and uses an LLM to generate structured, traceable test cases grounded strictly in that context.

## How it works

1. **Ingest** – load a requirement document (Markdown) from `data/requirements/`.
2. **Chunk** – split the document into overlapping text chunks using `langchain-text-splitters`.
3. **Embed** – encode each chunk into a vector using a `sentence-transformers` model (`all-MiniLM-L6-v2`).
4. **Index** – build an in-memory FAISS vector index (`IndexFlatL2`) over the chunk embeddings.
5. **Retrieve** – embed the user's query and fetch the top-k most similar chunks from the index.
6. **Generate** – send the retrieved context to an OpenAI model, which returns structured test cases (via a Pydantic schema) that cite their source requirement chunks.

## Project structure

```
qa-testcase-rag/
├── data/
│   └── requirements/       # Requirement/spec documents (Markdown)
│       ├── login.md
│       ├── checkout.md
│       └── payment.md
├── src/
│   ├── ingest.py            # Load requirement documents
│   ├── chunk.py             # Split documents into chunks
│   ├── embed.py             # Create embeddings for chunks
│   ├── vector_store.py      # Build/query the FAISS index
│   ├── retrieve.py          # Retrieve relevant chunks for a query
│   ├── generate.py          # Generate structured test cases via LLM
│   ├── schemas.py           # Pydantic schemas for test case output
│   ├── config.py            # Configuration (WIP)
│   └── rag.py               # End-to-end pipeline entry point
├── requirements.txt
└── .env                     # Local environment variables (not committed)
```

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

   `.env` is already listed in `.gitignore` and should never be committed.

## Usage

Run the end-to-end pipeline from the project root:

```bash
python src/rag.py
```

> **Note:** Modules under `src/` import each other directly (e.g. `from ingest import load_markdown_file`) rather than as a package, so they must be run as scripts (`python src/rag.py`), not as a module (`python -m src.rag`) — the latter will raise `ModuleNotFoundError`.

By default, `rag.py` loads `data/requirements/login.md` and generates test cases for the query defined in its `__main__` block. To generate test cases for a different question, edit the `query` variable in `src/rag.py`.

Each module (`ingest.py`, `chunk.py`, `embed.py`, `vector_store.py`, `retrieve.py`) can also be run individually (`python src/<module>.py`) to inspect that step of the pipeline in isolation.

## Output format

Test cases are returned as structured `TestCase` objects (see `src/schemas.py`) with the following fields:

- `test_case_id`
- `title`
- `scenario`
- `preconditions`
- `steps`
- `expected_result`
- `priority`
- `source` – the requirement chunk(s) the test case is derived from

## Notes

- The LLM is instructed (see `src/generate.py`) to generate test cases using only information present in the retrieved requirement context, and to cite the source for every test case, in order to avoid hallucinated business rules or UI behavior.
- Embeddings and the vector index are rebuilt in memory on every run; there is no persisted vector store yet.
