import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import harness  # noqa: E402
from eval.harness import PASS, Metrics, record  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def warm_model():
    """Load the model once so no layer's timing includes the ~35s cold start."""
    from agent.nodes import get_llm

    t0 = time.time()
    get_llm().invoke("ok")
    cold = time.time() - t0
    record("0-setup", "cold start (model load)", PASS, Metrics(wall_s=round(cold, 2)),
           note="excluded from layer timings")
    yield


@pytest.fixture(scope="session", autouse=True)
def frozen_index():
    """Fail loudly if the index changed mid-campaign, rather than silently skewing recall."""
    fp = harness.index_fingerprint()
    assert fp["chunks_jsonl"] > 0, "chunks.jsonl is empty - build the index first"
    assert fp["chroma_embeddings"] > 0, "Chroma collection is empty - build the index first"
    yield fp


@pytest.fixture(scope="session")
def qa_set():
    return harness.load_qa_set()


def pytest_sessionfinish(session, exitstatus):
    if harness.RESULTS:
        path = harness.write_results()
        print(f"\n\nResults table written to {path}")
        print("\n" + harness.summary_text())
