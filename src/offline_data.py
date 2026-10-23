"""Deterministic CSV loading for offline, timestamp-aware Metro evaluation.

The source timestamps are naive local clock times. No timezone or forecast
horizon is inferred. Duplicate weather observations are reduced to the first
valid-temperature source row per timestamp, after checking target agreement
across *all* rows. No distribution-derived filtering is applied.
"""

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = frozenset(
    {"date_time", "holiday", "temp", "clouds_all", "weather_main", "traffic_volume"}
)


def encode_holiday(value: str | None) -> int:
    """Distinguish the CSV's literal ``None`` from named holidays."""
    return int(value is not None and value.strip() not in {"", "None"})


def _hour_category(hour: int) -> str:
    return (
        "Late Night",
        "Early Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
    )[hour // 4]


@dataclass(frozen=True)
class TrafficRecord:
    date_time: datetime
    traffic_volume: float
    temp: float  # Celsius, converted from the source's Kelvin values.
    clouds_all: float
    weather_main: str
    holiday: int

    def features(self) -> dict[str, float | int | str]:
        """Return independent feature data using the existing transformer names."""
        return {
            "temp": self.temp,
            "clouds_all": self.clouds_all,
            "weather_main": self.weather_main,
            "holiday": self.holiday,
            "weekday": self.date_time.weekday(),
            "hour": self.date_time.hour,
            "month": self.date_time.month,
            "hour_category": _hour_category(self.date_time.hour),
        }


@dataclass(frozen=True)
class LoadDiagnostics:
    raw_count: int
    invalid_temp_dropped: int
    duplicate_extras: int
    distinct_timestamps: int


@dataclass(frozen=True)
class TemporalPartitions:
    train: tuple[TrafficRecord, ...]
    validation: tuple[TrafficRecord, ...]
    test: tuple[TrafficRecord, ...]


def _finite_number(value: str, name: str, line_number: int) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"CSV line {line_number}: {name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"CSV line {line_number}: {name} must be finite")
    return number


def load_csv(path: str | Path) -> tuple[tuple[TrafficRecord, ...], LoadDiagnostics]:
    """Load, validate and sort the source CSV without network access or fitting.

    Nonpositive Kelvin temperatures are physically invalid and are excluded;
    malformed and nonfinite numeric values are errors. Zero traffic is retained.
    ``duplicate_extras`` counts valid-temperature rows removed by deduplication,
    so raw_count == invalid_temp_dropped + duplicate_extras + distinct_timestamps.
    """
    records: dict[datetime, TrafficRecord] = {}
    targets: dict[datetime, float] = {}
    raw_count = invalid_temp_dropped = duplicate_extras = 0
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if columns is None:
            raise ValueError("CSV has no header")
        if len(set(columns)) != len(columns):
            raise ValueError("CSV has duplicate column names")
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            raw_count += 1
            line_number = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV line {line_number}: row width differs from header")
            try:
                date_time = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError(
                    f"CSV line {line_number}: date_time must be YYYY-MM-DD HH:MM:SS"
                ) from exc
            target = _finite_number(row["traffic_volume"], "traffic_volume", line_number)
            if target < 0:
                raise ValueError(f"CSV line {line_number}: traffic_volume must be nonnegative")
            if date_time in targets and targets[date_time] != target:
                raise ValueError(
                    f"CSV line {line_number}: conflicting traffic_volume for {date_time}"
                )
            targets[date_time] = target
            kelvin = _finite_number(row["temp"], "temp", line_number)
            clouds = _finite_number(row["clouds_all"], "clouds_all", line_number)
            if not 0 <= clouds <= 100:
                raise ValueError(f"CSV line {line_number}: clouds_all must be between 0 and 100")
            weather = row["weather_main"].strip()
            if not weather:
                raise ValueError(f"CSV line {line_number}: weather_main must not be empty")
            if kelvin <= 0:
                invalid_temp_dropped += 1
                continue
            if date_time in records:
                duplicate_extras += 1
                continue
            records[date_time] = TrafficRecord(
                date_time=date_time,
                traffic_volume=target,
                temp=kelvin - 273.15,
                clouds_all=clouds,
                weather_main=weather,
                holiday=encode_holiday(row["holiday"]),
            )
    ordered = tuple(records[key] for key in sorted(records))
    if not ordered:
        raise ValueError("CSV contains no usable records")
    return ordered, LoadDiagnostics(
        raw_count, invalid_temp_dropped, duplicate_extras, len(ordered)
    )


def _cutoff(value: str) -> datetime:
    try:
        cutoff = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Cutoffs must be ISO dates or naive ISO datetimes") from exc
    if cutoff.tzinfo is not None:
        raise ValueError("Cutoffs must be naive, matching the source timestamps")
    return cutoff


def partition_records(
    records: Iterable[TrafficRecord],
    validation_start: str = "2017-01-01",
    test_start: str = "2018-01-01",
) -> TemporalPartitions:
    """Partition by calendar cutoffs without inspecting target values.

    The default train partition ends before 2017, validation is calendar 2017,
    and test starts in 2018. Test may be empty for a development-only fixture;
    both train and validation must contain records. Cutoffs are not learned.
    """
    validation_cutoff, test_cutoff = _cutoff(validation_start), _cutoff(test_start)
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must precede test_start")
    supplied = tuple(records)
    timestamps = [record.date_time for record in supplied]
    if any(timestamp.tzinfo is not None for timestamp in timestamps):
        raise ValueError("Records must have naive source timestamps")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Records must have unique timestamps before partitioning")
    ordered = tuple(sorted(supplied, key=lambda record: record.date_time))
    train = tuple(record for record in ordered if record.date_time < validation_cutoff)
    validation = tuple(
        record
        for record in ordered
        if validation_cutoff <= record.date_time < test_cutoff
    )
    test = tuple(record for record in ordered if record.date_time >= test_cutoff)
    if not train or not validation:
        raise ValueError("Train and validation partitions must both be nonempty")
    return TemporalPartitions(train, validation, test)
