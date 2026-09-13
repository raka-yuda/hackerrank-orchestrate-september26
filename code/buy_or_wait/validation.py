from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import re

from .data import Dataset
from .planner import Decision, round_money


EXPECTED_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_decisions(dataset: Dataset, decisions: list[Decision]) -> None:
    if len(decisions) != len(dataset.requests):
        raise ValueError("decision count does not match request count")
    if [decision.request_id for decision in decisions] != [request.request_id for request in dataset.requests]:
        raise ValueError("decision request IDs or order do not match input")

    for decision in decisions:
        request = dataset.request_by_id[decision.request_id]
        profile = dataset.profile_by_user[request.user_id]
        if not Decimal("0") <= decision.amount_safe_to_pay <= request.requested_amount:
            raise ValueError(f"unsafe amount bounds for {request.request_id}")
        if decision.affordability_status not in VALID_STATUSES:
            raise ValueError(f"invalid affordability status for {request.request_id}")
        if decision.recommended_payment_method not in VALID_METHODS:
            raise ValueError(f"invalid payment method for {request.request_id}")
        expected_status_methods = {
            "affordable_now": {"full_payment"},
            "affordable_later": {"wait"},
            "not_affordable": {"not_recommended"},
            "affordable_with_plan": {"full_payment", "partial_payment", "installments"},
        }
        if decision.recommended_payment_method not in expected_status_methods[decision.affordability_status]:
            raise ValueError(f"status/method mismatch for {request.request_id}")
        if decision.affordability_status == "affordable_now" and decision.earliest_date_for_full_payment != request.request_date:
            raise ValueError(f"affordable_now date mismatch for {request.request_id}")
        if decision.earliest_date_for_full_payment is not None and not (
            request.request_date
            <= decision.earliest_date_for_full_payment
            <= request.request_date + timedelta(days=90)
        ):
            raise ValueError(f"invalid earliest full-payment date for {request.request_id}")
        if not decision.decision_explanation.strip():
            raise ValueError(f"blank explanation for {request.request_id}")
        parsed_plan: list[tuple[date, Decimal]] = []
        if decision.payment_plan != "none":
            previous: date | None = None
            for item in decision.payment_plan.split("|"):
                raw_date, raw_amount = item.split(":", 1)
                payment_date = date.fromisoformat(raw_date)
                amount = Decimal(raw_amount)
                if amount <= 0:
                    raise ValueError(f"non-positive payment for {request.request_id}")
                if previous is not None and payment_date < previous:
                    raise ValueError(f"non-chronological plan for {request.request_id}")
                previous = payment_date
                parsed_plan.append((payment_date, amount))
        elif decision.recommended_payment_method != "not_recommended":
            raise ValueError(f"missing payment plan for {request.request_id}")

        method = decision.recommended_payment_method
        if method in {"full_payment", "partial_payment", "installments"} and method not in profile.payment_methods:
            raise ValueError(f"payment method is not accepted for {request.request_id}")
        if method == "wait" and "full_payment" not in profile.payment_methods:
            raise ValueError(f"wait requires accepted full payment for {request.request_id}")
        if parsed_plan and parsed_plan[0][0] < request.request_date:
            raise ValueError(f"plan starts before request date for {request.request_id}")
        if parsed_plan and parsed_plan[-1][0] > request.desired_completion_date:
            raise ValueError(f"plan misses completion deadline for {request.request_id}")
        if method in {"full_payment", "wait"}:
            if len(parsed_plan) != 1 or parsed_plan[0][1] != request.requested_amount:
                raise ValueError(f"invalid single-payment plan for {request.request_id}")
        elif method == "partial_payment":
            if (
                not request.allows_partial_payment
                or len(parsed_plan) != 2
                or parsed_plan[0] != (request.request_date, round_money(decision.amount_safe_to_pay))
                or decision.earliest_date_for_full_payment is None
                or parsed_plan[1][0] != decision.earliest_date_for_full_payment
                or sum((amount for _, amount in parsed_plan), Decimal("0")) != request.requested_amount
            ):
                raise ValueError(f"invalid partial-payment plan for {request.request_id}")
        elif method == "installments":
            supplied = {
                tuple(
                    (
                        option.first_payment_date
                        + timedelta(days=index * (option.payment_frequency_days or 0)),
                        option.payment_amount,
                    )
                    for index in range(option.number_of_payments)
                )
                for option in dataset.options_by_request.get(request.request_id, [])
                if option.payment_method == "installments"
            }
            if tuple(parsed_plan) not in supplied:
                raise ValueError(f"installment plan is not a supplied option for {request.request_id}")
            if profile.max_installment_months is None or len(parsed_plan) > profile.max_installment_months:
                raise ValueError(f"installment duration is not accepted for {request.request_id}")
        elif method == "not_recommended" and parsed_plan:
            raise ValueError(f"not_recommended must not include payments for {request.request_id}")

        changes = decision.spending_changes_needed.split("|") if decision.spending_changes_needed != "none" else []
        if len(changes) > 3:
            raise ValueError(f"too many spending changes for {request.request_id}")
        seen_events: set[str] = set()
        for change in changes:
            match = re.fullmatch(r"(stop|reduce_to):([^:]+)(?::([0-9]+(?:\.[0-9]+)?))?", change)
            if match is None:
                raise ValueError(f"malformed spending change for {request.request_id}")
            kind, event_id, raw_amount = match.groups()
            event = dataset.event_by_id.get(event_id)
            if event is None or event.user_id != request.user_id or event_id in seen_events:
                raise ValueError(f"invalid spending event for {request.request_id}")
            seen_events.add(event_id)
            if kind == "stop":
                if (
                    raw_amount is not None
                    or event.category not in profile.stoppable_categories
                    or event.flexibility not in {"stoppable", "reducible_or_stoppable"}
                ):
                    raise ValueError(f"ineligible stop action for {request.request_id}")
            else:
                if (
                    raw_amount is None
                    or event.category not in profile.reducible_categories
                    or event.flexibility not in {"reducible", "reducible_or_stoppable"}
                    or event.minimum_allowed_amount is None
                    or Decimal(raw_amount) != event.minimum_allowed_amount
                ):
                    raise ValueError(f"ineligible reduction for {request.request_id}")
        if method == "not_recommended" and changes:
            raise ValueError(f"not_recommended must not include spending changes for {request.request_id}")
