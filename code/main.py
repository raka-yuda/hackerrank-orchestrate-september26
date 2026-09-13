from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from buy_or_wait import DecisionEngine
from buy_or_wait.data import Dataset
from buy_or_wait.evidence import EvidenceResolver
from buy_or_wait.validation import EXPECTED_COLUMNS, validate_decisions


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "dataset").is_dir() else SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Buy or Wait? financial decisions")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--requests", default="requests.csv", choices=("requests.csv", "sample_requests.csv"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output.csv")
    parser.add_argument(
        "--evidence-provider",
        choices=("auto", "local", "gemini", "openai"),
        default=os.environ.get("EVIDENCE_PROVIDER", "auto"),
        help="auto prefers OpenAI, then Gemini, when its API key is set; otherwise local OCR",
    )
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
    )
    parser.add_argument(
        "--evidence-cache",
        type=Path,
        default=REPO_ROOT / ".cache" / "buy_or_wait" / "evidence.json",
    )
    parser.add_argument(
        "--usage-report",
        type=Path,
        default=SCRIPT_DIR / "evaluation" / "usage_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Dataset(args.dataset, request_file=args.requests)
    evidence = EvidenceResolver(
        dataset,
        provider=args.evidence_provider,
        model=(
            args.gemini_model
            if args.evidence_provider == "gemini"
            else args.openai_model if args.evidence_provider == "openai" else None
        ),
        cache_path=args.evidence_cache,
    )
    decisions = DecisionEngine(dataset, evidence=evidence).decide_all()
    validate_decisions(dataset, decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(decision.as_csv_row() for decision in decisions)
    evidence.write_usage_report(args.usage_report, len(decisions))
    print(f"Wrote {len(decisions)} decisions to {args.output}")
    print(f"Evidence provider: {evidence.provider}; usage report: {args.usage_report}")
    print(
        f"Model calls={evidence.stats.model_calls}; input tokens={evidence.stats.input_tokens}; "
        f"output tokens={evidence.stats.output_tokens}; cache hits={evidence.stats.cache_hits}; "
        f"local fallbacks={evidence.stats.validation_fallbacks}"
    )


if __name__ == "__main__":
    main()
