from __future__ import annotations

import argparse
import csv
import os
import sys
from decimal import Decimal
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent
CODE_ROOT = EVALUATION_DIR.parent
REPO_ROOT = CODE_ROOT.parent if (CODE_ROOT.parent / "dataset").is_dir() else CODE_ROOT
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from buy_or_wait import DecisionEngine  # noqa: E402
from buy_or_wait.data import Dataset  # noqa: E402
from buy_or_wait.evidence import EvidenceResolver  # noqa: E402


FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the 25 public examples")
    parser.add_argument(
        "--evidence-provider",
        choices=("auto", "local", "gemini", "openai"),
        default=os.environ.get("EVIDENCE_PROVIDER", "auto"),
    )
    parser.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"))
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
    evidence = EvidenceResolver(
        dataset,
        provider=args.evidence_provider,
        model=(
            args.gemini_model
            if args.evidence_provider == "gemini"
            else args.openai_model if args.evidence_provider == "openai" else None
        ),
        cache_path=REPO_ROOT / ".cache" / "buy_or_wait" / "evidence.json",
    )
    predictions = {
        decision.request_id: decision.as_csv_row()
        for decision in DecisionEngine(dataset, evidence=evidence).decide_all()
    }
    with (REPO_ROOT / "dataset" / "sample_requests.csv").open(encoding="utf-8-sig", newline="") as handle:
        expected = list(csv.DictReader(handle))

    exact = {field: 0 for field in FIELDS}
    absolute_error = Decimal("0")
    for row in expected:
        prediction = predictions[row["request_id"]]
        amount_error = abs(Decimal(prediction["amount_safe_to_pay"]) - Decimal(row["amount_safe_to_pay"]))
        absolute_error += amount_error
        print(
            f"{row['request_id']}: amount={prediction['amount_safe_to_pay']} "
            f"expected={row['amount_safe_to_pay']} error={amount_error}; "
            f"status={prediction['affordability_status']}/{row['affordability_status']}; "
            f"method={prediction['recommended_payment_method']}/{row['recommended_payment_method']}"
        )
        for field in FIELDS:
            if prediction[field] == row[field]:
                exact[field] += 1

    print("\nExact matches")
    for field, count in exact.items():
        print(f"  {field}: {count}/{len(expected)}")
    print(f"Amount MAE: {absolute_error / len(expected):.2f}")
    print(
        f"Evidence: {evidence.provider}; model calls={evidence.stats.model_calls}; "
        f"input tokens={evidence.stats.input_tokens}; output tokens={evidence.stats.output_tokens}; "
        f"cache hits={evidence.stats.cache_hits}; local fallbacks={evidence.stats.validation_fallbacks}"
    )


if __name__ == "__main__":
    main()
