from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent if (CODE_ROOT.parent / "dataset").is_dir() else CODE_ROOT
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from buy_or_wait.data import Dataset  # noqa: E402
from buy_or_wait.evidence import (  # noqa: E402
    EvidenceResolver,
    GeminiClient,
    OpenAIClient,
    UsageStats,
)
from buy_or_wait.forecast import ForecastBuilder, SpendingAction  # noqa: E402
from buy_or_wait.planner import Candidate, DecisionEngine  # noqa: E402
from buy_or_wait.validation import EXPECTED_COLUMNS, validate_decisions  # noqa: E402


class DecisionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = Dataset(REPO_ROOT / "dataset")

    def test_full_dataset_is_complete_and_valid(self) -> None:
        evidence = EvidenceResolver(self.dataset, provider="local")
        decisions = DecisionEngine(self.dataset, evidence=evidence).decide_all()
        self.assertEqual(250, len(decisions))
        validate_decisions(self.dataset, decisions)
        self.assertEqual(list(EXPECTED_COLUMNS), list(decisions[0].as_csv_row()))

    def test_decisions_are_deterministic(self) -> None:
        first_engine = DecisionEngine(
            self.dataset, evidence=EvidenceResolver(self.dataset, provider="local")
        )
        second_engine = DecisionEngine(
            self.dataset, evidence=EvidenceResolver(self.dataset, provider="local")
        )
        first = [row.as_csv_row() for row in first_engine.decide_all()]
        second = [row.as_csv_row() for row in second_engine.decide_all()]
        self.assertEqual(first, second)

    def test_serialized_partial_payment_plans_remain_safe(self) -> None:
        engine = DecisionEngine(
            self.dataset,
            evidence=EvidenceResolver(self.dataset, provider="local", cache_path=None),
        )
        checked = 0
        for request in self.dataset.requests:
            decision = engine.decide(request)
            if decision.recommended_payment_method != "partial_payment":
                continue
            payments = tuple(
                (date.fromisoformat(raw_date), Decimal(raw_amount))
                for raw_date, raw_amount in (
                    item.split(":", 1) for item in decision.payment_plan.split("|")
                )
            )
            forecast = engine.forecasts.build(request)
            self.assertTrue(
                forecast.simulate(payments).safe,
                f"serialized partial plan is unsafe for {request.request_id}",
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_variable_debit_estimate_uses_supported_history(self) -> None:
        amounts = [Decimal("80"), Decimal("120"), Decimal("100"), Decimal("140")]
        monthly = [SimpleNamespace(category="utilities")] * len(amounts)
        frequent = [SimpleNamespace(category="groceries")] * len(amounts)
        self.assertEqual(
            Decimal("110"), ForecastBuilder._amount_estimate(monthly, amounts, "debit")
        )
        self.assertEqual(
            Decimal("120"), ForecastBuilder._amount_estimate(frequent, amounts, "debit")
        )

    def test_first_frequent_budget_is_reserved_on_request_date(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        engine = DecisionEngine(
            samples, evidence=EvidenceResolver(samples, provider="local")
        )
        request, forecast = next(
            (request, forecast)
            for request in samples.requests
            for forecast in (engine.forecasts.build(request),)
            if any(
                flow.stream_key and flow.stream_key.startswith("frequent:")
                for flow in forecast.flows
            )
        )
        frequent_keys = {
            flow.stream_key
            for flow in forecast.flows
            if flow.stream_key and flow.stream_key.startswith("frequent:")
        }
        first_dates = {
            key: min(flow.flow_date for flow in forecast.flows if flow.stream_key == key)
            for key in frequent_keys
        }
        self.assertTrue(first_dates)
        self.assertTrue(all(value == request.request_date for value in first_dates.values()))

    def test_action_tie_break_prefers_smallest_sufficient_change(self) -> None:
        large = SpendingAction("reduce", "large", "large", Decimal("0"), Decimal("75"))
        small_a = SpendingAction("stop", "small_a", "small_a", None, Decimal("10"))
        small_b = SpendingAction("reduce", "small_b", "small_b", Decimal("1"), Decimal("25"))
        payment = ((self.dataset.requests[0].request_date, Decimal("100")),)
        larger_change = Candidate("full_payment", payment, (large,), Decimal("100"))
        smaller_change = Candidate("full_payment", payment, (small_a, small_b), Decimal("100"))
        self.assertLess(smaller_change.ranking_key(), larger_change.ranking_key())

    def test_base_salary_message_does_not_inflate_settled_cash_income(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        request = next(
            request
            for request in samples.requests
            if any(
                "gaji pokok" in message.text.casefold()
                or "base salary" in message.text.casefold()
                for message in samples.messages_by_user.get(request.user_id, [])
            )
        )
        forecast = DecisionEngine(
            samples, evidence=EvidenceResolver(samples, provider="local", cache_path=None)
        ).forecasts.build(request)
        income_amounts = {stream.amount for stream in forecast.streams if stream.direction == "credit"}
        settled_salary_amounts = {
            event.amount
            for event in samples.events_by_user[request.user_id]
            if event.status == "settled"
            and event.direction == "credit"
            and event.category == "salary"
            and event.amount is not None
        }
        self.assertTrue(income_amounts)
        self.assertTrue(income_amounts.issubset(settled_salary_amounts))

    def test_next_salary_message_applies_to_one_pay_cycle(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        request = next(
            request
            for request in samples.requests
            if any(
                "next salary" in message.text.casefold()
                or "next payslip" in message.text.casefold()
                for message in samples.messages_by_user.get(request.user_id, [])
            )
        )
        evidence = EvidenceResolver(samples, provider="local", cache_path=None).resolve(request)
        self.assertTrue(evidence.messages.income_one_cycle)

    def test_projection_relevant_image_amounts_resolve_locally(self) -> None:
        resolver = EvidenceResolver(self.dataset, provider="local")
        resolved = 0
        for request in self.dataset.requests:
            bundle = resolver.resolve(request)
            end = request.request_date + timedelta(days=90)
            for event in self.dataset.events_by_user.get(request.user_id, []):
                if (
                    event.amount is None
                    and event.status in {"pending", "scheduled"}
                    and request.request_date <= event.cash_date <= end
                ):
                    self.assertGreater(bundle.amount_for(event), 0)
                    resolved += 1
        self.assertGreater(resolved, 0)

    def test_gemini_port_parses_structured_output_and_usage_offline(self) -> None:
        stats = UsageStats(provider="gemini", model="test-model")

        def fake_transport(request, timeout):
            self.assertNotIn("unit-test-key", request.full_url)
            self.assertEqual(60.0, timeout)
            return json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps({"answer": "ok"})}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 15,
                    },
                }
            ).encode()

        client = GeminiClient(
            "unit-test-key",
            "test-model",
            stats,
            transport=fake_transport,
            minimum_interval=0,
        )
        self.assertEqual({"answer": "ok"}, client.generate([{"text": "test"}], {"type": "OBJECT"}))
        self.assertEqual((1, 12, 3, 15), (stats.model_calls, stats.input_tokens, stats.output_tokens, stats.total_tokens))

    def test_openai_port_parses_structured_output_and_usage_offline(self) -> None:
        stats = UsageStats(provider="openai", model="test-model")

        def fake_transport(request, timeout):
            self.assertEqual("https://api.openai.com/v1/responses", request.full_url)
            self.assertEqual("Bearer unit-test-key", request.get_header("Authorization"))
            self.assertEqual(90.0, timeout)
            payload = json.loads(request.data)
            self.assertFalse(payload["store"])
            self.assertEqual("input_text", payload["input"][0]["content"][0]["type"])
            output_schema = payload["text"]["format"]["schema"]
            self.assertEqual("object", output_schema["type"])
            self.assertFalse(output_schema["additionalProperties"])
            return json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps({"answer": "ok"})}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 14,
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 19,
                    },
                }
            ).encode()

        client = OpenAIClient(
            "unit-test-key",
            "test-model",
            stats,
            transport=fake_transport,
            minimum_interval=0,
        )
        schema = {
            "type": "OBJECT",
            "properties": {"answer": {"type": "STRING"}},
            "required": ["answer"],
        }
        self.assertEqual({"answer": "ok"}, client.generate([{"text": "test"}], schema))
        self.assertEqual(
            (1, 14, 5, 2, 19),
            (
                stats.model_calls,
                stats.input_tokens,
                stats.output_tokens,
                stats.thinking_tokens,
                stats.total_tokens,
            ),
        )

    def test_openai_port_converts_inline_image_to_data_url(self) -> None:
        stats = UsageStats(provider="openai", model="test-model")

        def fake_transport(request, _timeout):
            payload = json.loads(request.data)
            image = payload["input"][0]["content"][1]
            self.assertEqual("input_image", image["type"])
            self.assertEqual("data:image/png;base64,dGVzdA==", image["image_url"])
            return json.dumps(
                {
                    "output_text": json.dumps({"answer": "ok"}),
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
            ).encode()

        client = OpenAIClient(
            "unit-test-key",
            "test-model",
            stats,
            transport=fake_transport,
            minimum_interval=0,
        )
        schema = {
            "type": "OBJECT",
            "properties": {"answer": {"type": "STRING"}},
            "required": ["answer"],
        }
        result = client.generate(
            [
                {"text": "test"},
                {"inline_data": {"mime_type": "image/png", "data": "dGVzdA=="}},
            ],
            schema,
        )
        self.assertEqual({"answer": "ok"}, result)

    def test_usage_report_defaults_to_standard_pinned_model_cost(self) -> None:
        resolver = EvidenceResolver(
            self.dataset,
            provider="openai",
            model="gpt-4.1-mini-2025-04-14",
            api_key="unit-test-key",
            cache_path=None,
            minimum_interval=0,
        )
        resolver.stats.model_calls = 12
        resolver.stats.input_tokens = 26891
        resolver.stats.output_tokens = 14288
        resolver.stats.total_tokens = 41179
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            report = Path(directory) / "usage_report.md"
            resolver.write_usage_report(report, 250)
            text = report.read_text(encoding="utf-8")
        self.assertIn(
            "| Estimated model cost | USD 0.033617 | USD 0.00013447 |",
            text,
        )

    def test_evidence_interface_uses_injected_gemini_adapter(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        request = next(
            request
            for request in samples.requests
            if any(
                event.amount is None and event.status in {"pending", "scheduled"}
                for event in samples.events_by_user.get(request.user_id, [])
            )
        )
        event = next(
            event
            for event in samples.events_by_user[request.user_id]
            if event.amount is None and event.status in {"pending", "scheduled"}
        )

        def fake_transport(request, _timeout):
            payload = json.loads(request.data)
            properties = payload["generationConfig"]["responseSchema"]["properties"]
            value = (
                {
                    "amount": "1",
                    "currency": event.currency,
                    "selected_label": "validated total",
                    "confidence": 0.98,
                }
                if "amount" in properties
                else {"users": []}
            )
            return json.dumps(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(value)}]}}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 12,
                    },
                }
            ).encode()

        resolver = EvidenceResolver(
            samples,
            provider="gemini",
            api_key="unit-test-key",
            transport=fake_transport,
            minimum_interval=0,
        )
        bundle = resolver.resolve(request)
        self.assertEqual(Decimal("1"), bundle.amount_for(event))
        self.assertEqual("gemini", bundle.image_amounts[event.event_id].provider)
        self.assertGreaterEqual(resolver.stats.model_calls, 2)

    def test_evidence_interface_uses_injected_openai_adapter(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        request = next(
            request
            for request in samples.requests
            if any(
                event.amount is None and event.status in {"pending", "scheduled"}
                for event in samples.events_by_user.get(request.user_id, [])
            )
        )
        event = next(
            event
            for event in samples.events_by_user[request.user_id]
            if event.amount is None and event.status in {"pending", "scheduled"}
        )

        def fake_transport(request, _timeout):
            payload = json.loads(request.data)
            properties = payload["text"]["format"]["schema"]["properties"]
            value = (
                {
                    "amount": "1",
                    "currency": event.currency,
                    "selected_label": "validated total",
                    "confidence": 0.98,
                }
                if "amount" in properties
                else {"users": []}
            )
            return json.dumps(
                {
                    "output_text": json.dumps(value),
                    "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                }
            ).encode()

        resolver = EvidenceResolver(
            samples,
            provider="openai",
            model="test-model",
            api_key="unit-test-key",
            transport=fake_transport,
            minimum_interval=0,
        )
        bundle = resolver.resolve(request)
        self.assertEqual(Decimal("1"), bundle.amount_for(event))
        self.assertEqual("openai", bundle.image_amounts[event.event_id].provider)
        self.assertGreaterEqual(resolver.stats.model_calls, 2)

    def test_public_example_uses_same_validated_engine(self) -> None:
        samples = Dataset(REPO_ROOT / "dataset", request_file="sample_requests.csv")
        evidence = EvidenceResolver(samples, provider="local")
        request = samples.requests[0]
        decision = DecisionEngine(samples, evidence=evidence).decide(request)
        row = decision.as_csv_row()
        self.assertEqual(request.request_id, row["request_id"])
        self.assertGreaterEqual(Decimal(row["amount_safe_to_pay"]), 0)
        self.assertLessEqual(Decimal(row["amount_safe_to_pay"]), request.requested_amount)


if __name__ == "__main__":
    unittest.main()
