"""Option type for representing optional values.

The Option type represents an optional value: every Option is either Some and contains
a value, or Nothing, and does not. This is similar to OCaml's option type.

IMPORTANT: Option types MUST be handled exhaustively using either:
1. Python's native match-case statement
2. Pyrethrin's match() function

Other methods like map(), and_then(), etc. are intentionally NOT provided
because they bypass exhaustive checking.

Usage:
    from pyrethrin import Some, Nothing, Option, match, returns_option

    @returns_option
    def find_user(user_id: str) -> Option[User]:
        user = db.get(user_id)
        if user is None:
            return Nothing()
        return Some(user)

    # Pattern matching (REQUIRED for exhaustive handling)
    result = find_user("123")
    match result:
        case Some(user):
            print(f"Found: {user.name}")
        case Nothing():
            print("User not found")

    # Or using Pyrethrin's match() function:
    match(find_user, "123")({
        Some: lambda user: print(f"Found: {user.name}"),
        Nothing: lambda: print("User not found"),
    })
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class UnwrapNothingError(Exception):
    """Raised when attempting to unwrap a Nothing value."""

    def __init__(self, message: str = "called `unwrap()` on a `Nothing` value"):
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class Some(Generic[T]):
    """Contains a value of type T.

    Use pattern matching (match-case) to extract the value exhaustively.
    """

    value: T
    __match_args__ = ("value",)

    def is_some(self) -> bool:
        """Returns True."""
        return True

    def is_nothing(self) -> bool:
        """Returns False."""
        return False

    def __repr__(self) -> str:
        return f"Some({self.value!r})"


@dataclass(frozen=True, slots=True)
class Nothing(Generic[T]):
    """Represents the absence of a value.

    Use pattern matching (match-case) to handle this case exhaustively.
    """

    __match_args__ = ()

    def is_some(self) -> bool:
        """Returns False."""
        return False

    def is_nothing(self) -> bool:
        """Returns True."""
        return True

    def __repr__(self) -> str:
        return "Nothing()"

    def __eq__(self, other: object) -> bool:
        """All Nothing instances are equal."""
        return isinstance(other, Nothing)

    def __hash__(self) -> int:
        """All Nothing instances have the same hash."""
        return hash(Nothing)


# Type alias for Option
Option = Some[T] | Nothing[T]


# Type variable for error in to_result
E = TypeVar("E", bound=BaseException)
