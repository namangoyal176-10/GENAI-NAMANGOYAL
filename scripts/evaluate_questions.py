"""Evaluation script to execute the five mandatory assignment questions through the RAG pipeline."""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVALUATION_DIR
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateQuestions")

ASSIGNMENT_QUESTIONS = [
    "What are the three main types of inspection?",
    "What does a maintenance factor of two mean?",
    "What is the purchase price of this gas turbine?",
    "What is the melting point of titanium?",
    "What is the recommended combustion-inspection interval in hours?",
]


def run_evaluation(output_path: Path = EVALUATION_DIR / "results.json"):
    """Runs the 5 assignment questions through the RAG pipeline and saves results to JSON."""
    print("=" * 80)
    print("GAS TURBINE MAINTENANCE ASSISTANT - ASSIGNMENT EVALUATION")
    print("=" * 80)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = RAGPipeline()

    # Pre-flight check
    setup_err = pipeline.check_setup()
    if setup_err:
        print(f"\n[ERROR] {setup_err}")
        sys.exit(1)

    results = []

    for idx, question in enumerate(ASSIGNMENT_QUESTIONS, start=1):
        print(f"\n[{idx}/5] Evaluating: \"{question}\"")
        start_t = time.time()
        response = pipeline.query(question)
        elapsed = time.time() - start_t

        print(f"Elapsed: {elapsed:.2f}s | Supported: {response.is_supported}")
        print("-" * 80)
        print(response.formatted_response)
        print("-" * 80)

        record = {
            "question_id": f"Q{idx}",
            "question": question,
            "answer": response.answer,
            "is_supported": response.is_supported,
            "reference": response.reference if response.is_supported else None,
            "pdf_page": response.pdf_page if response.is_supported else None,
            "printed_page": response.printed_page if response.is_supported else None,
            "section": response.section if response.is_supported else None,
            "evidence": response.evidence if response.is_supported else None,
            "formatted_output": response.formatted_response,
            "retrieval_distance": response.best_distance,
            "retrieval_similarity": response.best_similarity,
            "validation_reason": response.validation_reason,
            "elapsed_seconds": round(elapsed, 3),
        }
        results.append(record)

    # Save to evaluation/results.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "document": "GER-3620-P.pdf",
                    "total_questions": len(ASSIGNMENT_QUESTIONS),
                    "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nEvaluation completed. Results saved to: {output_path.resolve()}\n")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
