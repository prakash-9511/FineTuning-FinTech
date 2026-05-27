import argparse
import subprocess
import sys


TEST_CASES = [
    {
        "instruction": "Explain EBITDA and why it is not the same as free cash flow.",
        "input": "",
        "keywords": ["earnings", "interest", "taxes", "depreciation", "amortization", "cash flow"],
    },
    {
        "instruction": "What is equity dilution?",
        "input": "",
        "keywords": ["new shares", "ownership", "shareholders"],
    },
    {
        "instruction": "Explain Kubernetes for microservices.",
        "input": "",
        "keywords": ["containers", "scaling", "deployment", "services"],
    },
    {
        "instruction": "How does a vector database help RAG?",
        "input": "",
        "keywords": ["embeddings", "semantic", "retrieval", "context"],
    },
]


def generate(model_name: str, adapter_path: str, instruction: str, context: str) -> str:
    command = [
        sys.executable,
        "scripts/inference.py",
        "--model_name",
        model_name,
        "--adapter_path",
        adapter_path,
        "--instruction",
        instruction,
        "--input",
        context,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers with simple keyword coverage.")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", required=True)
    args = parser.parse_args()

    total_keywords = 0
    matched_keywords = 0

    for index, case in enumerate(TEST_CASES, start=1):
        answer = generate(args.model_name, args.adapter_path, case["instruction"], case["input"])
        answer_lower = answer.lower()
        matches = [keyword for keyword in case["keywords"] if keyword.lower() in answer_lower]
        total_keywords += len(case["keywords"])
        matched_keywords += len(matches)

        print(f"\nCase {index}: {case['instruction']}")
        print(f"Matched keywords: {matches}")
        print(f"Answer: {answer}")

    score = matched_keywords / total_keywords if total_keywords else 0.0
    print(f"\nKeyword coverage: {matched_keywords}/{total_keywords} = {score:.1%}")


if __name__ == "__main__":
    main()
