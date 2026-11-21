import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import GeminiModel

from src.retriever import build_retriever

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


retriever = build_retriever()

test_cases = []

for g in data:
    retrieved = retriever.invoke(g['question'])
    retrieved_context = [doc.page_content for doc in retrieved]

    test_cases.append(
        LLMTestCase(
            input=g['question'],
            expected_output=g['answer'],
            retrieval_context=retrieved_context,
            actual_output="(generator not evaluate in this run)"
        )
    )


metrices = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)
]

evaluate(
    test_cases=test_cases,
    metrics=metrices,
    hyperparameters= {
        "retriever": "base_k5",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 600,
        "chunk_overlap": 60,
        "tok_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_DATA_PATH
    }
)