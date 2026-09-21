from sentence_transformers import SentenceTransformer

from ingest import load_markdown_file
from chunk import create_chunks
from embed import create_embeddings
from vector_store import build_index
from retrieve import retrieve
from generate import generate_test_cases


MODEL_NAME = "all-MiniLM-L6-v2"


def run_rag(user_query: str):

    # 1. Load requirement
    document = load_markdown_file(
        "data/requirements/login.md"
    )

    # 2. Chunk
    chunks = create_chunks(document)

    # 3. Create embeddings
    embeddings = create_embeddings(chunks)

    # 4. Build vector index
    index = build_index(embeddings)

    # 5. Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 6. Retrieve relevant context
    results = retrieve(
        query=user_query,
        chunks=chunks,
        index=index,
        model=model,
        top_k=3
    )

    # 7. Generate test cases
    test_cases = generate_test_cases(
        user_request=user_query,
        retrieved_context=results
    )

    return test_cases


if __name__ == "__main__":

    query = "Generate test cases for password expiry after 90 days."

    result = run_rag(query)

    for test_case in result.test_cases:

        print("\n==============================")
        print(test_case.test_case_id)
        print("==============================")

        print("Title:")
        print(test_case.title)

        print("\nScenario:")
        print(test_case.scenario)

        print("\nPreconditions:")
        for item in test_case.preconditions:
            print("-", item)

        print("\nSteps:")
        for step in test_case.steps:
            print("-", step)

        print("\nExpected Result:")
        print(test_case.expected_result)

        print("\nPriority:")
        print(test_case.priority)

        print("\nSource:")
        for source in test_case.source:
            print("-", source)