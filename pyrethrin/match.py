from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pyrethrin.decorators import ExhaustiveSignature, OptionSignature
from pyrethrin.exceptions import ExhaustivenessError
from pyrethrin.option import Nothing, Some
from pyrethrin.result import Err, Ok

T = TypeVar("T")


class MatchBuilder(Generic[T]):
    """Match builder for @raises decorated functions (Result type)."""

    __slots__ = ("_fn", "_fn_name", "_args", "_kwargs", "_signature")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        signature: ExhaustiveSignature,
    ):
        self._fn = fn
        self._fn_name = getattr(fn, "__name__", str(fn))
        self._args = args
        self._kwargs = kwargs
        self._signature = signature

    def __call__(self, handlers: dict[type, Callable[..., T]]) -> T:
        self._validate(handlers)
        result = self._fn(*self._args, **self._kwargs)
        return self._dispatch(result, handlers)

    def _validate(self, handlers: dict[type, Callable[..., Any]]) -> None:
        provided = set(handlers.keys())
        declared = self._signature.exceptions
        required = declared | {Ok}

        if Ok not in provided:
            raise ExhaustivenessError(
                "Non-exhaustive match",
                func_name=self._fn_name,
                missing=[Ok],
                declared=list(declared),
                provided=list(provided),
            )

        missing = required - provided
        if missing:
            raise ExhaustivenessError(
                "Non-exhaustive match",
                func_name=self._fn_name,
                missing=list(missing),
                declared=list(declared),
                provided=list(provided),
            )

        extra = provided - required
        if extra:
            raise ExhaustivenessError(
                "Unexpected handlers in match",
                func_name=self._fn_name,
                extra=list(extra),
                declared=list(declared),
                provided=list(provided),
            )

    def _dispatch(self, result: Ok[Any] | Err[Any], handlers: dict[type, Callable[..., T]]) -> T:
        match result:
            case Ok(value):
                return handlers[Ok](value)
            case Err(error):
                for exc_type, handler in handlers.items():
                    if exc_type is not Ok and isinstance(error, exc_type):
                        return handler(error)
        raise RuntimeError("Unreachable: no handler matched")


class OptionMatchBuilder(Generic[T]):
    """Match builder for @returns_option decorated functions (Option type)."""

    __slots__ = ("_fn", "_fn_name", "_args", "_kwargs", "_signature")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        signature: OptionSignature,
    ):
        self._fn = fn
        self._fn_name = getattr(fn, "__name__", str(fn))
        self._args = args
        self._kwargs = kwargs
        self._signature = signature

    def __call__(self, handlers: dict[type, Callable[..., T]]) -> T:
        self._validate(handlers)
        result = self._fn(*self._args, **self._kwargs)
        return self._dispatch(result, handlers)

    def _validate(self, handlers: dict[type, Callable[..., Any]]) -> None:
        provided = set(handlers.keys())
        required = {Some, Nothing}

        if Some not in provided:
            raise ExhaustivenessError(
                "Non-exhaustive match",
                func_name=self._fn_name,
                missing=[Some],
                declared=[Some, Nothing],
                provided=list(provided),
            )

        if Nothing not in provided:
            raise ExhaustivenessError(
                "Non-exhaustive match",
                func_name=self._fn_name,
                missing=[Nothing],
                declared=[Some, Nothing],
                provided=list(provided),
            )

        extra = provided - required
        if extra:
            raise ExhaustivenessError(
                "Unexpected handlers in match",
                func_name=self._fn_name,
                extra=list(extra),
                declared=[Some, Nothing],
                provided=list(provided),
            )

    def _dispatch(
        self, result: Some[Any] | Nothing[Any], handlers: dict[type, Callable[..., T]]
    ) -> T:
        match result:
            case Some(value):
                return handlers[Some](value)
            case Nothing():
                return handlers[Nothing]()
        raise RuntimeError("Unreachable: no handler matched")


def match(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> MatchBuilder[Any] | OptionMatchBuilder[Any]:
    """
    Create a match builder for exhaustive handling of Result or Option types.

    For @raises decorated functions (Result type):
        match(get_user, user_id)({
            Ok: lambda user: user.name,
            UserNotFound: lambda e: "Unknown",
            InvalidUserId: lambda e: "Invalid",
        })

    For @returns_option decorated functions (Option type):
        match(find_user, user_id)({
            Some: lambda user: user.name,
            Nothing: lambda: "Unknown",
        })
    """
    # Check for @raises decorated function (Result type)
    if hasattr(fn, "__pyrethrin_signature__"):
        return MatchBuilder(fn, args, kwargs, fn.__pyrethrin_signature__)  # type: ignore[attr-defined]

    # Check for @returns_option decorated function (Option type)
    if hasattr(fn, "__pyrethrin_option_signature__"):
        return OptionMatchBuilder(fn, args, kwargs, fn.__pyrethrin_option_signature__)  # type: ignore[attr-defined]

    # Neither decorator found
    fn_name = getattr(fn, "__name__", str(fn))
    raise TypeError(
        f"match() requires a @raises or @returns_option decorated function, got `{fn_name}`\n\n"
        f"  For functions that return Result (can raise exceptions):\n\n"
        f"    @raises(SomeError, AnotherError)\n"
        f"    def {fn_name}(...):\n"
        f"        ...\n\n"
        f"  For functions that return Option (optional values):\n\n"
        f"    @returns_option\n"
        f"    def {fn_name}(...) -> Option[T]:\n"
        f"        ..."
    )
