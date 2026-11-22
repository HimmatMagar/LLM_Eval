import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval.models import GeminiModel

from src.pipeline import Pipeline

load_dotenv()

GOLDEN_DATA_PATH = "./goldens/golden_data.json"
THRESHOLD = 0.2
JUDGE_MODEL = GeminiModel(
    model="gemini-3.8-flash",
    temperature=0
)

### Load the golden dataset
with open(GOLDEN_DATA_PATH) as f:
    data = json.load(f)


rag = Pipeline()

test_cases = []

for g in data:
    result = rag.invoke(g['question'])

    test_cases.append(
        LLMTestCase(
            input=g['question'],
            actual_output=result['ans']
        )
    )


metrices = [
    ToxicityMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, strict_mode=False)
]

evaluate(
    test_cases=test_cases,
    metrics=metrices
)