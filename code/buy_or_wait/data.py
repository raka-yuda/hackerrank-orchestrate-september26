from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Iterable


ZERO = Decimal("0")


def money(value: str | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if value == "":
        raise ValueError("blank monetary value")
    return Decimal(str(value))


def split_pipe(value: str) -> frozenset[str]:
    return frozenset(part for part in value.split("|") if part)


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    current_balance: Decimal
    minimum_balance: Decimal
    priorities: frozenset[str]
    protected_categories: frozenset[str]
    reducible_categories: frozenset[str]
    stoppable_categories: frozenset[str]
    payment_methods: frozenset[str]
    max_installment_months: int | None


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: str
    source_type: str
    text: str


@dataclass(frozen=True)
class ImageLink:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Dataset:
    """Loads participant-facing data and exposes validated lookup indexes."""

    def __init__(self, dataset_dir: Path, request_file: str = "requests.csv") -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.profiles = self._load_profiles()
        self.requests = self._load_requests(request_file)
        self.events = self._load_events()
        self.options = self._load_options()
        self.messages = self._load_messages()
        self.images = self._load_images()
        self.rates = self._load_rates()

        self.profile_by_user = {row.user_id: row for row in self.profiles}
        self.request_by_id = {row.request_id: row for row in self.requests}
        self.event_by_id = {row.event_id: row for row in self.events}
        self.events_by_user = self._group(self.events, "user_id")
        self.options_by_request = self._group(self.options, "request_id")
        self.messages_by_user = self._group(self.messages, "user_id")
        self.image_by_event = {row.related_event_id: row for row in self.images}
        self._validate()

    @staticmethod
    def _group(rows: Iterable[object], attribute: str) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(getattr(row, attribute), []).append(row)
        return grouped

    def _load_profiles(self) -> list[Profile]:
        result = []
        for row in _rows(self.dataset_dir / "financial_profiles.csv"):
            result.append(
                Profile(
                    user_id=row["user_id"],
                    home_currency=row["home_currency"],
                    current_balance=money(row["current_available_balance"]),
                    minimum_balance=money(row["minimum_balance_to_keep"]),
                    priorities=split_pipe(row["financial_priorities"]),
                    protected_categories=split_pipe(row["expense_categories_to_protect"]),
                    reducible_categories=split_pipe(row["expense_categories_user_is_willing_to_reduce"]),
                    stoppable_categories=split_pipe(row["expense_categories_user_is_willing_to_stop"]),
                    payment_methods=split_pipe(row["payment_methods_user_will_consider"]),
                    max_installment_months=int(row["max_installment_months"]) if row["max_installment_months"] else None,
                )
            )
        return result

    def _load_requests(self, filename: str) -> list[Request]:
        result = []
        for row in _rows(self.dataset_dir / filename):
            result.append(
                Request(
                    request_id=row["request_id"],
                    user_id=row["user_id"],
                    request_date=date.fromisoformat(row["request_date"]),
                    request_type=row["request_type"],
                    requested_amount=money(row["requested_amount"]),
                    desired_completion_date=date.fromisoformat(row["desired_completion_date"]),
                    allows_partial_payment=row["allows_partial_payment"].lower() == "true",
                    request_text=row["request_text"],
                )
            )
        return result

    def _load_events(self) -> list[Event]:
        result = []
        for row in _rows(self.dataset_dir / "financial_events.csv"):
            result.append(
                Event(
                    event_id=row["event_id"],
                    user_id=row["user_id"],
                    event_type=row["event_type"],
                    description=row["description"],
                    category=row["category"],
                    direction=row["direction"],
                    amount=money(row["amount"]) if row["amount"] else None,
                    currency=row["currency"],
                    event_date=date.fromisoformat(row["event_date"]),
                    settlement_date=date.fromisoformat(row["settlement_date"]) if row["settlement_date"] else None,
                    status=row["status"],
                    linked_event_id=row["linked_event_id"] or None,
                    flexibility=row["flexibility"],
                    minimum_allowed_amount=money(row["minimum_allowed_amount"]) if row["minimum_allowed_amount"] else None,
                )
            )
        return result

    def _load_options(self) -> list[PaymentOption]:
        result = []
        for row in _rows(self.dataset_dir / "request_payment_options.csv"):
            result.append(
                PaymentOption(
                    payment_option_id=row["payment_option_id"],
                    request_id=row["request_id"],
                    payment_method=row["payment_method"],
                    payment_amount=money(row["payment_amount"]),
                    number_of_payments=int(row["number_of_payments"]),
                    first_payment_date=date.fromisoformat(row["first_payment_date"]),
                    payment_frequency_days=int(row["payment_frequency_days"]) if row["payment_frequency_days"] else None,
                    financing_fee=money(row["financing_fee"]),
                    total_payable_amount=money(row["total_payable_amount"]),
                )
            )
        return result

    def _load_messages(self) -> list[Message]:
        return [
            Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=row["request_id"] or None,
                related_event_id=row["related_event_id"] or None,
                sent_at=row["sent_at"],
                source_type=row["source_type"],
                text=row["message_text"],
            )
            for row in _rows(self.dataset_dir / "messages.csv")
        ]

    def _load_images(self) -> list[ImageLink]:
        return [ImageLink(**row) for row in _rows(self.dataset_dir / "images.csv")]

    def _load_rates(self) -> dict[tuple[date, str, str], Decimal]:
        return {
            (date.fromisoformat(row["rate_date"]), row["from_currency"], row["to_currency"]): money(row["rate"])
            for row in _rows(self.dataset_dir / "exchange_rates.csv")
        }

    def _validate(self) -> None:
        if len(self.profile_by_user) != len(self.profiles):
            raise ValueError("duplicate user_id in financial_profiles.csv")
        if len(self.request_by_id) != len(self.requests):
            raise ValueError("duplicate request_id")
        if len(self.event_by_id) != len(self.events):
            raise ValueError("duplicate event_id")
        for request in self.requests:
            if request.user_id not in self.profile_by_user:
                raise ValueError(f"unknown user for {request.request_id}")
        for image in self.images:
            event = self.event_by_id.get(image.related_event_id)
            if event is None or event.user_id != image.user_id:
                raise ValueError(f"invalid image/event link for {image.image_id}")

    def convert(self, amount: Decimal, from_currency: str, to_currency: str, on_date: date) -> Decimal:
        if from_currency == to_currency:
            return amount
        key = (on_date, from_currency, to_currency)
        try:
            return amount * self.rates[key]
        except KeyError as exc:
            raise ValueError(
                f"missing exchange rate for {on_date}: {from_currency}->{to_currency}"
            ) from exc
