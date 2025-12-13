"""Result type for representing success or failure.

The Result type represents either success (Ok) or failure (Err). This is similar
to Rust's Result type and OCaml's result type.

IMPORTANT: Result types MUST be handled exhaustively using either:
1. Python's native match-case statement
2. Pyrethrin's match() function

Other methods like map(), unwrap(), etc. are intentionally NOT provided
because they bypass exhaustive checking.

Usage:
    from pyrethrin import Ok, Err, Result, match, raises

    @raises(ValueError, KeyError)
    def get_user(user_id: str) -> dict:
        if not user_id:
            raise ValueError("empty id")
        return {"id": user_id}

    # Pattern matching (REQUIRED for exhaustive handling)
    result = get_user("123")
    match result:
        case Ok(user):
            print(f"Found: {user}")
        case Err(ValueError() as e):
            print(f"Invalid: {e}")
        case Err(KeyError() as e):
            print(f"Not found: {e}")

    # Or using Pyrethrin's match() function:
    match(get_user, "123")({
        Ok: lambda user: print(f"Found: {user}"),
        ValueError: lambda e: print(f"Invalid: {e}"),
        KeyError: lambda e: print(f"Not found: {e}"),
    })
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Represents a successful result containing a value.

    Use pattern matching (match-case) to extract the value exhaustively.
    """

    value: T
    __match_args__ = ("value",)

    def is_ok(self) -> bool:
        """Returns True."""
        return True

    def is_err(self) -> bool:
        """Returns False."""
        return False

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Represents a failed result containing an error.

    Use pattern matching (match-case) to handle this case exhaustively.
    """

    error: E
    __match_args__ = ("error",)

    def is_ok(self) -> bool:
        """Returns False."""
        return False

    def is_err(self) -> bool:
        """Returns True."""
        return True

    def __repr__(self) -> str:
        return f"Err({self.error!r})"


Result = Ok[T] | Err[E]
