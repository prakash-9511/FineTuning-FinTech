import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {"domain", "instruction", "input", "output"}
ALLOWED_DOMAINS = {"finance", "tech"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} is missing fields: {sorted(missing)}")
            if row["domain"] not in ALLOWED_DOMAINS:
                raise ValueError(
                    f"{path}:{line_number} has unsupported domain {row['domain']!r}; "
                    f"expected one of {sorted(ALLOWED_DOMAINS)}"
                )
            if not row["instruction"].strip() or not row["output"].strip():
                raise ValueError(f"{path}:{line_number} must include non-empty instruction and output")
            rows.append(row)
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    counts = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
    print(f"{name}: {len(rows)} examples")
    for domain, count in sorted(counts.items()):
        print(f"  - {domain}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate JSONL instruction-tuning datasets.")
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/validation.jsonl"))
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    validation_rows = read_jsonl(args.validation)

    summarize("train", train_rows)
    summarize("validation", validation_rows)
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
