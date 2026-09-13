from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from itertools import combinations

from .data import Dataset, PaymentOption, Request, ZERO
from .evidence import EvidenceResolver
from .forecast import Forecast, ForecastBuilder, SpendingAction


CENT = Decimal("0.01")


def round_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def floor_money(value: Decimal) -> Decimal:
    """Return the largest currency-cent amount that does not exceed value."""
    return value.quantize(CENT, rounding=ROUND_DOWN)


def format_money(value: Decimal) -> str:
    rounded = round_money(value)
    if rounded == rounded.to_integral_value():
        return str(rounded.to_integral_value())
    return f"{rounded:.2f}"


def format_numeric(value: Decimal) -> str:
    return format(value.normalize(), "f") if value else "0"


def option_number(option_id: str | None) -> int:
    if option_id is None:
        return 10**9
    try:
        return int(option_id.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def human_date(value: date) -> str:
    """Format without platform-specific strftime flags."""
    return f"{value.day} {value.strftime('%B %Y')}"


@dataclass(frozen=True)
class Candidate:
    method: str
    payments: tuple[tuple[date, Decimal], ...]
    actions: tuple[SpendingAction, ...]
    total_payable: Decimal
    option_id: str | None = None

    @property
    def completion_date(self) -> date:
        return self.payments[-1][0]

    @property
    def start_date(self) -> date:
        return self.payments[0][0]

    def ranking_key(self) -> tuple:
        return (
            1 if self.actions else 0,
            round_money(self.total_payable),
            self.start_date,
            len(self.payments),
            sum((action.estimated_savings for action in self.actions), ZERO),
            len(self.actions),
            option_number(self.option_id),
        )


@dataclass(frozen=True)
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": format_numeric(round_money(self.amount_safe_to_pay)),
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": self.earliest_date_for_full_payment.isoformat() if self.earliest_date_for_full_payment else "",
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }


class DecisionEngine:
    def __init__(self, dataset: Dataset, evidence: EvidenceResolver | None = None) -> None:
        self.dataset = dataset
        self.evidence = evidence or EvidenceResolver(dataset)
        self.forecasts = ForecastBuilder(dataset, self.evidence)

    @staticmethod
    def _installment_payments(option: PaymentOption) -> tuple[tuple[date, Decimal], ...]:
        frequency = option.payment_frequency_days or 0
        return tuple(
            (option.first_payment_date + timedelta(days=index * frequency), option.payment_amount)
            for index in range(option.number_of_payments)
        )

    @staticmethod
    def _valid_action_sets(actions: list[SpendingAction]):
        limited = actions[:12]
        for count in range(1, min(3, len(limited)) + 1):
            for selected in combinations(limited, count):
                if len({action.stream_key for action in selected}) != len(selected):
                    continue
                yield tuple(sorted(selected, key=lambda action: action.event_id))

    def _base_candidates(
        self,
        request: Request,
        forecast: Forecast,
        safe_today: Decimal,
        earliest_full: date | None,
        actions: tuple[SpendingAction, ...] = (),
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        methods = forecast.profile.payment_methods

        full_today = ((request.request_date, request.requested_amount),)
        if "full_payment" in methods and forecast.simulate(full_today, actions).safe:
            candidates.append(
                Candidate(
                    method="full_payment",
                    payments=full_today,
                    actions=actions,
                    total_payable=request.requested_amount,
                )
            )

        if not actions and "full_payment" in methods and earliest_full is not None:
            if request.request_date < earliest_full <= request.desired_completion_date:
                wait_plan = ((earliest_full, request.requested_amount),)
                if forecast.simulate(wait_plan).safe:
                    candidates.append(
                        Candidate(
                            method="wait",
                            payments=wait_plan,
                            actions=(),
                            total_payable=request.requested_amount,
                        )
                    )

        if (
            not actions
            and request.allows_partial_payment
            and "partial_payment" in methods
            and ZERO < safe_today < request.requested_amount
            and earliest_full is not None
            and earliest_full <= request.desired_completion_date
        ):
            remainder = request.requested_amount - safe_today
            partial_plan = (
                (request.request_date, safe_today),
                (earliest_full, remainder),
            )
            if forecast.simulate(partial_plan).safe:
                candidates.append(
                    Candidate(
                        method="partial_payment",
                        payments=partial_plan,
                        actions=(),
                        total_payable=request.requested_amount,
                    )
                )

        if "installments" in methods:
            for option in self.dataset.options_by_request.get(request.request_id, []):
                if option.payment_method != "installments":
                    continue
                if forecast.profile.max_installment_months is None:
                    continue
                if option.number_of_payments > forecast.profile.max_installment_months:
                    continue
                payments = self._installment_payments(option)
                if payments[-1][0] > request.desired_completion_date:
                    continue
                if forecast.simulate(payments, actions).safe:
                    candidates.append(
                        Candidate(
                            method="installments",
                            payments=payments,
                            actions=actions,
                            total_payable=option.total_payable_amount,
                            option_id=option.payment_option_id,
                        )
                    )
        return candidates

    @staticmethod
    def _payment_plan(candidate: Candidate | None) -> str:
        if candidate is None:
            return "none"
        return "|".join(
            f"{payment_date.isoformat()}:{format_money(amount)}"
            for payment_date, amount in candidate.payments
        )

    @staticmethod
    def _status(candidate: Candidate | None, request: Request) -> str:
        if candidate is None:
            return "not_affordable"
        if candidate.method == "full_payment" and not candidate.actions and candidate.start_date == request.request_date:
            return "affordable_now"
        if candidate.method == "wait":
            return "affordable_later"
        return "affordable_with_plan"

    @staticmethod
    def _explanation(
        request: Request,
        forecast: Forecast,
        candidate: Candidate | None,
        simulation_minimum: Decimal,
        safe_today: Decimal,
    ) -> str:
        currency = forecast.profile.home_currency
        minimum = format_money(forecast.profile.minimum_balance)
        requested = format_money(request.requested_amount)
        if candidate is None:
            if safe_today > ZERO:
                return (
                    f"Do not proceed with the {currency} {requested} request. Although {currency} "
                    f"{format_money(safe_today)} is safe today, no eligible option completes it by "
                    f"{human_date(request.desired_completion_date)} while protecting the {currency} {minimum} minimum."
                )
            return (
                f"Do not make this payment by {human_date(request.desired_completion_date)}. "
                f"No eligible option keeps the {currency} {minimum} minimum protected."
            )

        action_prefix = ""
        if candidate.actions:
            rendered = []
            for action in candidate.actions:
                if action.kind == "stop":
                    rendered.append(f"stop {action.event_id}")
                else:
                    rendered.append(f"reduce {action.event_id} to {currency} {format_money(action.new_amount or ZERO)}")
            action_prefix = ", then ".join(rendered).capitalize() + ", then "

        if candidate.method == "full_payment":
            lead = f"{action_prefix}pay {currency} {requested} today."
        elif candidate.method == "wait":
            lead = (
                f"Wait until {human_date(candidate.start_date)}, then pay "
                f"{currency} {requested} in full."
            )
        elif candidate.method == "partial_payment":
            first = candidate.payments[0][1]
            second_date, second = candidate.payments[1]
            lead = (
                f"Pay {currency} {format_money(first)} today and the remaining {currency} "
                f"{format_money(second)} on {human_date(second_date)}."
            )
        else:
            first_date, first_amount = candidate.payments[0]
            lead = (
                f"Use {len(candidate.payments)} installments of {currency} {format_money(first_amount)}, "
                f"starting {human_date(first_date)}."
            )
        return f"{lead[0].upper()}{lead[1:]} This keeps at least {currency} {format_money(simulation_minimum)} available."

    def decide(self, request: Request) -> Decision:
        forecast = self.forecasts.build(request)
        # Plans and CSV output use currency cents. Flooring here prevents a
        # serialized payment from rounding above the amount proven safe.
        safe_today = floor_money(forecast.safe_amount_today())
        earliest_full = forecast.earliest_safe_full_payment_date(request.requested_amount)
        candidates = self._base_candidates(request, forecast, safe_today, earliest_full)
        if not candidates:
            for action_set in self._valid_action_sets(forecast.eligible_actions()):
                candidates.extend(
                    self._base_candidates(
                        request,
                        forecast,
                        safe_today,
                        earliest_full,
                        action_set,
                    )
                )
        candidate = min(candidates, key=lambda item: item.ranking_key()) if candidates else None
        simulation = forecast.simulate(candidate.payments, candidate.actions) if candidate else forecast.simulate()
        changes = (
            "|".join(action.render(format_money) for action in candidate.actions)
            if candidate and candidate.actions
            else "none"
        )
        return Decision(
            request_id=request.request_id,
            amount_safe_to_pay=safe_today,
            affordability_status=self._status(candidate, request),
            recommended_payment_method=candidate.method if candidate else "not_recommended",
            payment_plan=self._payment_plan(candidate),
            earliest_date_for_full_payment=earliest_full,
            spending_changes_needed=changes,
            decision_explanation=self._explanation(
                request,
                forecast,
                candidate,
                simulation.minimum_projected_balance,
                safe_today,
            ),
        )

    def decide_all(self) -> list[Decision]:
        return [self.decide(request) for request in self.dataset.requests]
