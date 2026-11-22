import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.pipeline import Pipeline

load_dotenv()

GOLDEN_DATA_PATH = "./goldens/golden_data.json"
THRESHOLD = 0.75
JUDGE_MODEL = 'gpt-4o-mini'


with open(GOLDEN_DATA_PATH) as f:
    data = json.load(f)

rag = Pipeline()

test_cases = []

for g in data:
    result = rag.invoke(g['question'])

    test_cases.append(
        LLMTestCase(
            input=g['question'],
            actual_output=result['answer'],
            expected_output=g['answer']
        )
    )


correctness = GEval(
    name="Correctness",
    # evaluation_steps=[
    #     "Compare the actual output against the key facts in expected output",

    # ],
    criteria="Determin whether the actual output is factually based on the expected output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=THRESHOLD,
    model= JUDGE_MODEL,
    strict_mode=False
)

evaluate(test_cases=test_cases, metrics=[correctness])