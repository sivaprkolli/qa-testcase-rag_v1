from pydantic import BaseModel
from typing import List


class TestCase(BaseModel):
    test_case_id: str
    title: str
    scenario: str
    preconditions: List[str]
    steps: List[str]
    expected_result: str
    priority: str
    source: List[str]


class TestCaseResponse(BaseModel):
    test_cases: List[TestCase]