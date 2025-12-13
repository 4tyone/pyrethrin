from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, Generic, ParamSpec, TypeVar

from pyrethrin.decorators import ExhaustiveSignature, _check_caller_exhaustiveness
from pyrethrin.exceptions import ExhaustivenessError, UndeclaredExceptionError
from pyrethrin.result import Err, Ok, Result

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def async_raises(
    *exc_types: type[BaseException],
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[Result[R, BaseException]]]]:
    signature = ExhaustiveSignature(*exc_types)

    def decorator(
        fn: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[Result[R, BaseException]]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[R, BaseException]:
            # Run static analysis on the caller's code
            _check_caller_exhaustiveness(fn.__name__, exc_types)

            try:
                result = await fn(*args, **kwargs)
                if isinstance(result, (Ok, Err)):
                    return result
                return Ok(result)
            except BaseException as e:
                if any(isinstance(e, exc) for exc in exc_types):
                    return Err(e)
                raise UndeclaredExceptionError(
                    fn=fn.__name__,
                    got=type(e).__name__,
                    declared=[t.__name__ for t in exc_types],
                    original=e,
                ) from e

        wrapper.__pyrethrin_signature__ = signature  # type: ignore[attr-defined]
        wrapper.__pyrethrin_raises__ = frozenset(exc_types)  # type: ignore[attr-defined]
        return wrapper

    return decorator


class AsyncMatchBuilder(Generic[T]):
    __slots__ = ("_fn", "_fn_name", "_args", "_kwargs", "_signature")

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        signature: ExhaustiveSignature,
    ):
        self._fn = fn
        self._fn_name = getattr(fn, "__name__", str(fn))
        self._args = args
        self._kwargs = kwargs
        self._signature = signature

    async def __call__(self, handlers: dict[type, Callable[..., T]]) -> T:
        self._validate(handlers)
        result = await self._fn(*self._args, **self._kwargs)
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


def async_match(
    fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
) -> AsyncMatchBuilder[Any]:
    if not hasattr(fn, "__pyrethrin_signature__"):
        fn_name = getattr(fn, "__name__", str(fn))
        raise TypeError(
            f"async_match() requires a @async_raises decorated function, got `{fn_name}`\n\n"
            f"  To fix, add the @async_raises decorator to `{fn_name}`:\n\n"
            f"    @async_raises(SomeError, AnotherError)\n"
            f"    async def {fn_name}(...):\n"
            f"        ..."
        )
    return AsyncMatchBuilder(fn, args, kwargs, fn.__pyrethrin_signature__)  # type: ignore[attr-defined]
