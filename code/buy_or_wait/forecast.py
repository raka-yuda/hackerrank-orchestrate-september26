from __future__ import annotations

import calendar
import statistics
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal

from .data import Dataset, Event, Profile, Request, ZERO
from .evidence import EvidenceBundle, EvidenceResolver, MessageEvidence


FORECAST_DAYS = 90
FREQUENT_CATEGORIES = frozenset({"groceries", "transport", "dining"})
DISCRETIONARY_CATEGORIES = frozenset({"dining", "shopping", "entertainment"})
NON_RECURRING_TYPES = frozenset({"refund", "investment_purchase", "investment_sale", "investment_valuation"})
ONE_TIME_INCOME_WORDS = (
    "bonus",
    "commission",
    "arrears",
    "reimbursement",
    "prize",
    "investment sale",
)


def add_months(value: date, months: int = 1) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class RecurringStream:
    key: str
    source_event_id: str
    description: str
    category: str
    direction: str
    amount: Decimal
    currency: str
    next_date: date
    cadence_days: int | None
    monthly: bool
    flexibility: str
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True)
class CashFlow:
    flow_date: date
    amount: Decimal
    description: str
    category: str
    stream_key: str | None = None
    source_event_id: str | None = None


@dataclass(frozen=True)
class SpendingAction:
    kind: str
    event_id: str
    stream_key: str
    new_amount: Decimal | None
    estimated_savings: Decimal

    def render(self, formatter) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{formatter(self.new_amount or ZERO)}"


@dataclass(frozen=True)
class Simulation:
    safe: bool
    minimum_projected_balance: Decimal
    ending_balance: Decimal


class Forecast:
    def __init__(
        self,
        profile: Profile,
        request: Request,
        flows: list[CashFlow],
        streams: list[RecurringStream],
    ) -> None:
        self.profile = profile
        self.request = request
        self.flows = tuple(flows)
        self.streams = tuple(streams)

    def simulate(
        self,
        payments: tuple[tuple[date, Decimal], ...] = (),
        actions: tuple[SpendingAction, ...] = (),
    ) -> Simulation:
        stop_keys = {action.stream_key for action in actions if action.kind == "stop"}
        reductions = {
            action.stream_key: action.new_amount
            for action in actions
            if action.kind == "reduce" and action.new_amount is not None
        }
        flows: list[CashFlow] = []
        for flow in self.flows:
            if flow.stream_key in stop_keys:
                continue
            if flow.stream_key in reductions and flow.amount < ZERO:
                flow = replace(flow, amount=-(reductions[flow.stream_key] or ZERO))
            flows.append(flow)
        for payment_date, amount in payments:
            flows.append(
                CashFlow(
                    flow_date=payment_date,
                    amount=-amount,
                    description="recommended request payment",
                    category="request_payment",
                )
            )

        balance = self.profile.current_balance
        minimum = balance
        # Dated income is available on its settlement date. Ordinary cash flows
        # settle before a recommendation made for that date; request payments
        # are therefore processed last.
        flows.sort(
            key=lambda flow: (
                flow.flow_date,
                2 if flow.category == "request_payment" else (0 if flow.amount > ZERO else 1),
                flow.description,
            )
        )
        for flow in flows:
            balance += flow.amount
            minimum = min(minimum, balance)
        return Simulation(
            safe=minimum >= self.profile.minimum_balance,
            minimum_projected_balance=minimum,
            ending_balance=balance,
        )

    def safe_amount_today(self) -> Decimal:
        baseline = self.simulate()
        cushion = baseline.minimum_projected_balance - self.profile.minimum_balance
        return max(ZERO, min(self.request.requested_amount, cushion))

    def earliest_safe_full_payment_date(self, amount: Decimal) -> date | None:
        final_date = self.request.request_date + timedelta(days=FORECAST_DAYS)
        current = self.request.request_date
        while current <= final_date:
            if self.simulate(((current, amount),)).safe:
                return current
            current += timedelta(days=1)
        return None

    def eligible_actions(self) -> list[SpendingAction]:
        actions: list[SpendingAction] = []
        for stream in self.streams:
            if stream.direction != "debit":
                continue
            projected_count = sum(flow.stream_key == stream.key for flow in self.flows)
            if projected_count == 0:
                continue
            if (
                stream.category in self.profile.stoppable_categories
                and stream.flexibility in {"stoppable", "reducible_or_stoppable"}
            ):
                actions.append(
                    SpendingAction(
                        kind="stop",
                        event_id=stream.source_event_id,
                        stream_key=stream.key,
                        new_amount=None,
                        estimated_savings=stream.amount * projected_count,
                    )
                )
            if (
                stream.category in self.profile.reducible_categories
                and stream.flexibility in {"reducible", "reducible_or_stoppable"}
                and stream.minimum_allowed_amount is not None
                and stream.minimum_allowed_amount < stream.amount
            ):
                actions.append(
                    SpendingAction(
                        kind="reduce",
                        event_id=stream.source_event_id,
                        stream_key=stream.key,
                        new_amount=stream.minimum_allowed_amount,
                        estimated_savings=(stream.amount - stream.minimum_allowed_amount) * projected_count,
                    )
                )
        # Larger savings first makes bounded combination search deterministic.
        return sorted(actions, key=lambda action: (-action.estimated_savings, action.event_id, action.kind))


class ForecastBuilder:
    def __init__(self, dataset: Dataset, evidence: EvidenceResolver) -> None:
        self.dataset = dataset
        self.evidence = evidence

    def _home_amount(
        self,
        event: Event,
        bundle: EvidenceBundle,
        on_date: date | None = None,
    ) -> Decimal:
        amount = bundle.amount_for(event)
        profile = self.dataset.profile_by_user[event.user_id]
        return self.dataset.convert(amount, event.currency, profile.home_currency, on_date or event.cash_date)

    @staticmethod
    def _amount_estimate(events: list[Event], amounts: list[Decimal], direction: str) -> Decimal:
        recent = amounts[-3:]
        if max(recent) - min(recent) <= Decimal("0.01"):
            return recent[-1]
        if direction == "credit":
            # A single partial or corrected payroll credit should not become the
            # permanent salary forecast when the surrounding settled payroll
            # history supports the normal amount.
            return sorted(recent)[len(recent) // 2]
        if events and events[0].category in FREQUENT_CATEGORIES:
            # Frequent variable spending should react to recent behavior.
            return sum(recent, ZERO) / Decimal(len(recent))
        # Monthly streams have few observations, so use all recurrence-supported
        # history instead of letting one unusually cheap or expensive month
        # dominate the reserve.
        return sum(amounts, ZERO) / Decimal(len(amounts))

    @staticmethod
    def _stable_cadence(events: list[Event], max_days: int) -> int | None:
        if len(events) < 4:
            return None
        deltas = [
            (later.cash_date - earlier.cash_date).days
            for earlier, later in zip(events, events[1:])
            if (later.cash_date - earlier.cash_date).days > 0
        ]
        if len(deltas) < 3:
            return None
        recent = deltas[-8:]
        cadence = round(statistics.median(recent))
        consistent = sum(abs(delta - cadence) <= 1 for delta in recent)
        if cadence <= max_days and consistent >= max(3, len(recent) - 2):
            return cadence
        return None

    @staticmethod
    def _monthly(events: list[Event]) -> bool:
        if len(events) < 2:
            return False
        deltas = [(b.cash_date - a.cash_date).days for a, b in zip(events, events[1:])]
        recent = deltas[-4:]
        return sum(27 <= delta <= 32 for delta in recent) >= min(2, len(recent))

    @staticmethod
    def _include_discretionary(event: Event) -> bool:
        # Fixed discretionary purchases are historical behavior, not protected
        # commitments. Flexible recurring spending is retained in the baseline
        # so an explicit stop/reduction can be evaluated when the profile allows it.
        return not (
            event.category in DISCRETIONARY_CATEGORIES
            and event.flexibility == "fixed"
        )

    def _history(self, request: Request) -> list[Event]:
        return [
            event
            for event in self.dataset.events_by_user.get(request.user_id, [])
            if event.status == "settled"
            and event.direction != "non_cash"
            and event.cash_date <= request.request_date
        ]

    def _expense_streams(
        self,
        request: Request,
        history: list[Event],
        evidence: MessageEvidence,
        bundle: EvidenceBundle,
    ) -> list[RecurringStream]:
        end = request.request_date + timedelta(days=FORECAST_DAYS)
        streams: list[RecurringStream] = []
        used: set[str] = set()

        for category in FREQUENT_CATEGORIES:
            events = sorted(
                (
                    event
                    for event in history
                    if event.category == category
                    and event.direction == "debit"
                    and event.event_type == "expense"
                    and event.amount is not None
                    and self._include_discretionary(event)
                ),
                key=lambda event: event.cash_date,
            )
            cadence = self._stable_cadence(events, 21)
            if cadence is None:
                continue
            amounts = [self._home_amount(event, bundle) for event in events]
            amount = self._amount_estimate(events, amounts, "debit")
            next_date = events[-1].cash_date + timedelta(days=cadence)
            if next_date > end:
                continue
            source = events[-1]
            streams.append(
                RecurringStream(
                    key=f"frequent:{category}",
                    source_event_id=source.event_id,
                    description=f"recurring {category}",
                    category=category,
                    direction="debit",
                    amount=amount,
                    currency=self.dataset.profile_by_user[request.user_id].home_currency,
                    next_date=next_date,
                    cadence_days=cadence,
                    monthly=False,
                    flexibility=source.flexibility,
                    minimum_allowed_amount=source.minimum_allowed_amount,
                )
            )
            used.update(event.event_id for event in events)

        groups: dict[tuple[str, str], list[Event]] = {}
        for event in history:
            if (
                event.event_id in used
                or event.direction != "debit"
                or event.category in FREQUENT_CATEGORIES
                or event.event_type in NON_RECURRING_TYPES
                or event.amount is None
            ):
                continue
            groups.setdefault((event.description, event.currency), []).append(event)

        for (description, _currency), events in sorted(groups.items()):
            events.sort(key=lambda event: event.cash_date)
            if not self._monthly(events):
                continue
            amounts = [self._home_amount(event, bundle) for event in events]
            amount = self._amount_estimate(events, amounts, "debit")
            if events[-1].category == "rent" and evidence.rent_multiplier is not None:
                amount *= evidence.rent_multiplier
            next_date = add_months(events[-1].cash_date)
            if next_date > end:
                continue
            source = events[-1]
            streams.append(
                RecurringStream(
                    key=f"monthly:{description}:{source.currency}",
                    source_event_id=source.event_id,
                    description=description,
                    category=source.category,
                    direction="debit",
                    amount=amount,
                    currency=self.dataset.profile_by_user[request.user_id].home_currency,
                    next_date=next_date,
                    cadence_days=None,
                    monthly=True,
                    flexibility=source.flexibility,
                    minimum_allowed_amount=source.minimum_allowed_amount,
                )
            )
        return streams

    def _income_streams(
        self,
        request: Request,
        history: list[Event],
        evidence: MessageEvidence,
        bundle: EvidenceBundle,
    ) -> list[RecurringStream]:
        if evidence.stop_recurring_income:
            return []
        end = request.request_date + timedelta(days=FORECAST_DAYS)
        observed_salary = sorted(
            (
                event
                for event in history
                if event.direction == "credit" and event.category == "salary"
            ),
            key=lambda event: event.cash_date,
        )
        if observed_salary:
            latest_description = observed_salary[-1].description.casefold()
            terminal_salary = (
                any(term in latest_description for term in ("final", "last"))
                and any(term in latest_description for term in ("salary", "payroll", "employer"))
            ) or "termination" in latest_description
            if terminal_salary:
                return []
        salary_events = [
            event
            for event in history
            if event.direction == "credit"
            and event.category == "salary"
            and event.amount is not None
            and not any(word in event.description.lower() for word in ONE_TIME_INCOME_WORDS)
        ]

        groups: dict[tuple[str, str], list[Event]] = {}
        for event in salary_events:
            groups.setdefault((event.description, event.currency), []).append(event)

        streams: list[RecurringStream] = []
        for (description, _currency), events in sorted(groups.items()):
            events.sort(key=lambda event: event.cash_date)
            if len(events) < 2 or not self._monthly(events):
                continue
            amounts = [self._home_amount(event, bundle) for event in events]
            amount = self._amount_estimate(events, amounts, "credit")
            source = events[-1]
            next_date = add_months(source.cash_date)
            streams.append(
                RecurringStream(
                    key=f"income:{description}:{source.currency}",
                    source_event_id=source.event_id,
                    description=description,
                    category="salary",
                    direction="credit",
                    amount=amount,
                    currency=self.dataset.profile_by_user[request.user_id].home_currency,
                    next_date=next_date,
                    cadence_days=None,
                    monthly=True,
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                )
            )

        scheduled_salary = sorted(
            (
                event
                for event in self.dataset.events_by_user.get(request.user_id, [])
                if event.status == "scheduled"
                and event.direction == "credit"
                and event.category == "salary"
                and request.request_date <= event.cash_date <= end
            ),
            key=lambda event: event.cash_date,
        )
        if scheduled_salary:
            event = scheduled_salary[-1]
            amount = self._home_amount(event, bundle)
            streams = [
                RecurringStream(
                    key="income:confirmed_salary",
                    source_event_id=event.event_id,
                    description=event.description,
                    category="salary",
                    direction="credit",
                    amount=amount,
                    currency=self.dataset.profile_by_user[request.user_id].home_currency,
                    next_date=add_months(event.cash_date),
                    cadence_days=None,
                    monthly=True,
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                )
            ]

        if evidence.recurring_income_amount is not None and evidence.recurring_income_currency:
            effective_date = evidence.recurring_income_date
            if effective_date is None:
                effective_date = streams[0].next_date if streams else date(request.request_date.year, request.request_date.month, 15)
                if effective_date < request.request_date:
                    effective_date = add_months(effective_date)
            if evidence.income_one_cycle:
                # The message amends only the affected pay cycle. Preserve the
                # established salary stream from the following month onward.
                streams = [replace(stream, next_date=add_months(effective_date)) for stream in streams]
            else:
                home = self.dataset.profile_by_user[request.user_id].home_currency
                converted = self.dataset.convert(
                    evidence.recurring_income_amount,
                    evidence.recurring_income_currency,
                    home,
                    effective_date,
                )
                salary_messages = [
                    message.text.casefold()
                    for message in self.dataset.messages_by_user.get(request.user_id, [])
                    if message.source_type == "employer"
                ]
                headline_base_salary = any(
                    _term in text
                    for text in salary_messages
                    for _term in ("base salary", "gaji pokok")
                )
                explicit_cash_salary = any(
                    _term in text
                    for text in salary_messages
                    for _term in (
                        "net salary",
                        "take-home",
                        "credited to your account",
                        "masuk ke rekening",
                    )
                )
                if streams and headline_base_salary and not explicit_cash_salary:
                    # Base/gross salary is not necessarily spendable cash. When
                    # it conflicts with a lower settled payroll stream, retain
                    # the settled cash amount as the financially safer evidence.
                    converted = min(converted, streams[0].amount)
                source_id = streams[0].source_event_id if streams else "message_income"
                streams = [
                    RecurringStream(
                        key="income:message_confirmed",
                        source_event_id=source_id,
                        description="confirmed salary update",
                        category="salary",
                        direction="credit",
                        amount=converted,
                        currency=home,
                        next_date=effective_date,
                        cadence_days=None,
                        monthly=True,
                        flexibility="fixed",
                        minimum_allowed_amount=None,
                    )
                ]
        elif evidence.delayed_income_date is not None and streams:
            streams[0] = replace(streams[0], next_date=evidence.delayed_income_date)

        # A pending gig/service payout is not safe income. Do not suppress a
        # separately established calendar-month salary stream.
        pending_service_payout = any(
            message.source_type == "service_provider"
            and any(term in message.text.casefold() for term in ("payout", "payment", "pembayaran"))
            and any(term in message.text.casefold() for term in ("pending", "tertunda", "belum dapat ditarik"))
            for message in self.dataset.messages_by_user.get(request.user_id, [])
        )
        if evidence.suppress_unconfirmed_income and pending_service_payout:
            streams = []
        return [stream for stream in streams if stream.next_date <= end]

    @staticmethod
    def _expand_stream(stream: RecurringStream, start: date, end: date) -> list[CashFlow]:
        result = []
        current = stream.next_date
        while current <= end:
            if current >= start:
                signed = stream.amount if stream.direction == "credit" else -stream.amount
                result.append(
                    CashFlow(
                        flow_date=current,
                        amount=signed,
                        description=stream.description,
                        category=stream.category,
                        stream_key=stream.key,
                        source_event_id=stream.source_event_id,
                    )
                )
            current = add_months(current) if stream.monthly else current + timedelta(days=stream.cadence_days or 0)
        return result

    def build(self, request: Request) -> Forecast:
        profile = self.dataset.profile_by_user[request.user_id]
        history = self._history(request)
        bundle = self.evidence.resolve(request)
        message_evidence = bundle.messages
        streams = self._expense_streams(request, history, message_evidence, bundle)
        streams += self._income_streams(request, history, message_evidence, bundle)
        end = request.request_date + timedelta(days=FORECAST_DAYS)
        flows = [
            flow
            for stream in streams
            for flow in self._expand_stream(stream, request.request_date, end)
        ]

        # Groceries, transport, and dining are recurring budgets rather than
        # contractual settlement dates. Reserve the first projected occurrence
        # immediately so a purchase recommendation cannot rely on the user
        # postponing ordinary variable spending. Moving (rather than adding) the
        # flow preserves the inferred 90-day occurrence count.
        first_frequent: set[str] = set()
        front_loaded: list[CashFlow] = []
        for flow in sorted(flows, key=lambda item: (item.flow_date, item.stream_key or "")):
            if (
                flow.stream_key is not None
                and flow.stream_key.startswith("frequent:")
                and flow.stream_key not in first_frequent
            ):
                first_frequent.add(flow.stream_key)
                flow = replace(flow, flow_date=request.request_date)
            front_loaded.append(flow)
        flows = front_loaded

        # Explicit records override an inferred flow with the same category/date.
        for event in self.dataset.events_by_user.get(request.user_id, []):
            if event.status not in {"pending", "scheduled"} or event.direction == "non_cash":
                continue
            if not request.request_date <= event.cash_date <= end:
                continue
            if event.direction == "credit" and event.status == "pending":
                continue
            amount = self._home_amount(event, bundle)
            signed = amount if event.direction == "credit" else -amount
            if event.category == "salary" and event.direction == "credit":
                flows = [
                    flow
                    for flow in flows
                    if not (flow.flow_date == event.cash_date and flow.category == "salary")
                ]
            flows.append(
                CashFlow(
                    flow_date=event.cash_date,
                    amount=signed,
                    description=event.description,
                    category=event.category,
                    source_event_id=event.event_id,
                )
            )

        if (
            message_evidence.income_one_cycle
            and message_evidence.recurring_income_amount is not None
            and message_evidence.recurring_income_currency
        ):
            effective_date = message_evidence.recurring_income_date
            income_streams = [stream for stream in streams if stream.direction == "credit"]
            if effective_date is None and income_streams:
                effective_date = add_months(income_streams[0].next_date, -1)
            if effective_date is not None and request.request_date <= effective_date <= end:
                converted = self.dataset.convert(
                    message_evidence.recurring_income_amount,
                    message_evidence.recurring_income_currency,
                    profile.home_currency,
                    effective_date,
                )
                flows.append(
                    CashFlow(
                        flow_date=effective_date,
                        amount=converted,
                        description="confirmed temporary salary",
                        category="salary",
                    )
                )

        if (
            message_evidence.one_time_income_amount is not None
            and message_evidence.one_time_income_currency
            and message_evidence.one_time_income_date
            and request.request_date <= message_evidence.one_time_income_date <= end
        ):
            converted = self.dataset.convert(
                message_evidence.one_time_income_amount,
                message_evidence.one_time_income_currency,
                profile.home_currency,
                message_evidence.one_time_income_date,
            )
            flows.append(
                CashFlow(
                    flow_date=message_evidence.one_time_income_date,
                    amount=converted,
                    description="confirmed invoice payment",
                    category="salary",
                )
            )
        return Forecast(profile, request, flows, streams)
