import argparse
import json
from pathlib import Path

from rag_answer import TOP_K, load_index, retrieve


DEFAULT_DATASET_PATH = Path("evaluation/retrieval_questions.json")


def load_dataset(dataset_path: Path) -> list[dict]:
    """
    Load evaluation cases shaped like:
    [{"question": "...", "expected_files": ["path/one.py", ...]}, ...]
    """

    with dataset_path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{dataset_path} must contain a non-empty list.")

    for position, case in enumerate(cases, start=1):
        if not case.get("question") or not case.get("expected_files"):
            raise ValueError(
                f"Case {position} needs a 'question' and "
                "a non-empty 'expected_files' list."
            )

    return cases


def score_case(
    retrieved_files: list[str],
    expected_files: list[str],
) -> dict:
    """Compute file-level retrieval metrics for one question."""

    expected = set(expected_files)
    found = expected.intersection(retrieved_files)

    # Rank of the first relevant file, used for Mean Reciprocal Rank
    first_rank = next(
        (
            rank
            for rank, file_path in enumerate(retrieved_files, start=1)
            if file_path in expected
        ),
        None,
    )

    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(expected),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "missing": sorted(expected - found),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality against expected files.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Evaluation JSON file (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of retrieved files to score (default: {TOP_K}).",
    )

    arguments = parser.parse_args()

    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1.")

    return arguments


def main() -> None:
    arguments = parse_arguments()
    cases = load_dataset(arguments.dataset)
    index = load_index()

    results = []

    for position, case in enumerate(cases, start=1):
        best_indices, _ = retrieve(case["question"], index, arguments.top_k)
        retrieved_files = [
            index["chunks"][i]["file_path"] for i in best_indices
        ]

        result = score_case(retrieved_files, case["expected_files"])
        results.append(result)

        status = "HIT " if result["hit"] else "MISS"
        print(f"[{position}/{len(cases)}] {status} {case['question']}")
        print(f"   Recall: {result['recall']:.2f}")

        if result["missing"]:
            print(f"   Missing: {', '.join(result['missing'])}")

    k = arguments.top_k
    count = len(results)

    print("\nRetrieval evaluation:\n")
    print(f"Questions: {count}")
    print(f"Hit@{k}: {sum(r['hit'] for r in results) / count:.3f}")
    print(f"Recall@{k}: {sum(r['recall'] for r in results) / count:.3f}")
    print(f"MRR@{k}: {sum(r['reciprocal_rank'] for r in results) / count:.3f}")


if __name__ == "__main__":
    main()
