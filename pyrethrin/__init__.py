"""Pyrethrin: Exhaustive exception handling for Python."""

from pyrethrin.async_support import async_match, async_raises
from pyrethrin.decorators import raises, returns_option
from pyrethrin.exceptions import ExhaustivenessError, PyrethrinError, UndeclaredExceptionError
from pyrethrin.match import match
from pyrethrin.option import Nothing, Option, Some, UnwrapNothingError
from pyrethrin.result import Err, Ok, Result

__version__ = "0.1.0"

__all__ = [
    # Decorators
    "raises",
    "async_raises",
    "returns_option",
    # Match helpers
    "match",
    "async_match",
    # Result type
    "Ok",
    "Err",
    "Result",
    # Option type
    "Some",
    "Nothing",
    "Option",
    "UnwrapNothingError",
    # Exceptions
    "PyrethrinError",
    "ExhaustivenessError",
    "UndeclaredExceptionError",
]
