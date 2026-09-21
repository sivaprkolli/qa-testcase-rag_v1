import os

from dotenv import load_dotenv
from openai import OpenAI

from schemas import TestCaseResponse


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


SYSTEM_PROMPT = """
You are an experienced QA engineer.

Your task is to generate software test cases from
the provided requirements and acceptance criteria.

STRICT RULES:

1. Use ONLY information supported by the provided context.
2. Do not invent business rules.
3. Do not invent validations.
4. Do not invent error messages.
5. Do not invent UI behavior.
6. Do not assume requirements that are not provided.
7. Every test case must reference its source requirement
   or acceptance criterion.
8. Generate meaningful positive and negative scenarios
   when supported by the requirements.
9. Return the result using the requested structured schema.
"""


def generate_test_cases(
    user_request: str,
    retrieved_context: list[dict]
):

    context_parts = []

    for result in retrieved_context:

        chunk = result["chunk"]

        context_parts.append(
            f"""
SOURCE: {chunk["source"]}
CHUNK ID: {chunk["chunk_id"]}

CONTENT:
{chunk["text"]}
"""
        )

    context = "\n".join(context_parts)

    user_prompt = f"""
USER REQUEST:

{user_request}


RETRIEVED REQUIREMENT CONTEXT:

{context}


Generate test cases based strictly on the
retrieved requirement context.
"""

    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format=TestCaseResponse
    )

    return response.choices[0].message.parsed