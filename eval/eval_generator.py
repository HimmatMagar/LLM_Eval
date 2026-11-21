import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import GeminiModel

from src.generator import ask

load_dotenv()

GOLDEN_DATA_PATH = "./goldens/golden_data.json"
THRESHOLD = 0.75
JUDGE_MODEL = GeminiModel(
    model="gemini-3.8-flash",
    temperature=0
)

### Load the golden dataset
with open(GOLDEN_DATA_PATH) as f:
    data = json.load(f)


test_cases = []

for g in data:
    context = g['ideal_context']
    answer = ask(g['question'])

    test_cases.append(
        LLMTestCase(
            input=g['query'],
            expected_output=answer,
            retrieval_context=context,
        )
    )


metrices = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

evaluate(
    test_cases=test_cases,
    metrics=metrices
)