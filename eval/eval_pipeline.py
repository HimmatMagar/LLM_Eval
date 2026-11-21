import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

from src.pipeline import Pipeline

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL = ""
THRESHOLD = 0.7

with open(GOLDEN_PATH) as f:
    data = json.load(f)

rag = Pipeline()

test_cases = []
for g in data:
    result = rag.invoke(g['question'])

    test_cases.append(
        LLMTestCase(
            input=g['input'],
            actual_output=result['answer'],
            retrieval_context=result['context']
        )
    )


metrices = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

evaluate(test_cases=test_cases, metrics=metrices)