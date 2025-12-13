from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pyrethrin.exceptions import ExhaustivenessError


@contextmanager
def expect_exhaustive_error(
    *,
    missing: list[type] | None = None,
    extra: list[type] | None = None,
) -> Iterator[None]:
    try:
        yield
        raise AssertionError("Expected ExhaustivenessError was not raised")
    except ExhaustivenessError as e:
        if missing is not None:
            missing_set = set(missing)
            actual_missing = set(e.missing)
            if missing_set != actual_missing:
                raise AssertionError(
                    f"Expected missing={missing_set}, got missing={actual_missing}"
                ) from e
        if extra is not None:
            extra_set = set(extra)
            actual_extra = set(e.extra)
            if extra_set != actual_extra:
                raise AssertionError(f"Expected extra={extra_set}, got extra={actual_extra}") from e


def assert_raises_signature(
    fn: Callable[..., Any],
    expected: set[type[BaseException]],
) -> None:
    if not hasattr(fn, "__pyrethrin_raises__"):
        raise AssertionError(f"{fn.__name__} is not decorated with @raises")
    actual: frozenset[type[BaseException]] = fn.__pyrethrin_raises__  # type: ignore[attr-defined]
    if set(actual) != expected:
        raise AssertionError(f"Expected signature {expected}, got {set(actual)}")
