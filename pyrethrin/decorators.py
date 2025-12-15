from __future__ import annotations

import functools
import inspect
import linecache
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

from pyrethrin.exceptions import ExhaustivenessError, UndeclaredExceptionError
from pyrethrin.result import Err, Ok, Result

P = ParamSpec("P")
R = TypeVar("R")

# Cache for already-checked call sites to avoid re-running analysis
_checked_call_sites: set[tuple[str, int]] = set()

# Flag to disable static checking (useful for testing or performance)
PYRETHRIN_DISABLE_STATIC_CHECK = os.environ.get("PYRETHRIN_DISABLE_STATIC_CHECK", "").lower() in (
    "1",
    "true",
    "yes",
)


class ExhaustiveSignature:
    """Signature for @raises decorated functions (Result type)."""

    __slots__ = ("exceptions",)

    def __init__(self, *exc_types: type[BaseException]):
        self.exceptions: frozenset[type[BaseException]] = frozenset(exc_types)


class OptionSignature:
    """Signature for @returns_option decorated functions (Option type)."""

    __slots__ = ()

    def __init__(self):
        pass


def _get_pyrethrum_binary() -> Path | None:
    """Get path to bundled pyrethrum binary."""
    import platform

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            binary_name = "pyrethrum-darwin-arm64"
        else:
            binary_name = "pyrethrum-darwin-x64"
    elif system == "linux":
        binary_name = "pyrethrum-linux-x64"
    elif system == "windows":
        binary_name = "pyrethrum-windows-x64.exe"
    else:
        return None

    bin_dir = Path(__file__).parent / "bin"
    binary_path = bin_dir / binary_name
    if binary_path.exists():
        return binary_path
    return None


def _check_caller_exhaustiveness(
    func_name: str,
    exc_types: tuple[type[BaseException], ...] | None = None,
    signature_type: str = "raises",
) -> None:
    """
    Check if the caller's match statement is exhaustive.
    Analyzes the source file where the function was called from.

    Args:
        func_name: Name of the function being called
        exc_types: Exception types for @raises decorated functions (None for @returns_option)
        signature_type: Either "raises" (Result) or "option" (Option)
    """
    if PYRETHRIN_DISABLE_STATIC_CHECK:
        return

    pyrethrum = _get_pyrethrum_binary()
    if pyrethrum is None:
        return

    # Get caller's frame (skip internal pyrethrin frames to find user code)
    frame = inspect.currentframe()
    try:
        if frame is None:
            return

        # Walk up the stack until we find a frame outside the pyrethrin package
        pyrethrin_dir = str(Path(__file__).parent)
        caller_frame = frame.f_back
        while caller_frame is not None:
            frame_file = caller_frame.f_code.co_filename
            # Stop when we find a frame outside pyrethrin
            if not frame_file.startswith(pyrethrin_dir):
                break
            caller_frame = caller_frame.f_back

        if caller_frame is None:
            return

        filename = caller_frame.f_code.co_filename
        lineno = caller_frame.f_lineno

        # Skip if already checked this call site
        call_site = (filename, lineno)
        if call_site in _checked_call_sites:
            return
        _checked_call_sites.add(call_site)

        # Skip non-file sources (like <string>, <stdin>, etc.)
        if not filename or filename.startswith("<") or not os.path.isfile(filename):
            return

        # Build external signature for this function
        external_sig = {
            "name": func_name,
            "qualified_name": None,
            "declared_exceptions": [],
            "loc": {"file": "", "line": 0, "col": 0, "end_line": 0, "end_col": 0},
            "is_async": False,
            "signature_type": signature_type,
        }
        if exc_types:
            for exc in exc_types:
                module = exc.__module__
                name = exc.__name__
                if module and module != "builtins":
                    external_sig["declared_exceptions"].append({
                        "kind": "qualified",
                        "module": module,
                        "name": name,
                    })
                else:
                    external_sig["declared_exceptions"].append({
                        "kind": "name",
                        "name": name,
                    })

        # Read and analyze the source file with external signature
        from pyrethrin._ast_dump import dump_raw_ast_json

        try:
            json_data = dump_raw_ast_json(filename, external_signatures=[external_sig])
        except Exception:
            return

        try:
            proc = subprocess.run(
                [str(pyrethrum), "check", "--stdin", "-f", "json"],
                input=json_data,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return

        if proc.returncode == 0:
            return

        # Parse errors and raise for ones related to our function
        import json

        try:
            output = json.loads(proc.stdout)
        except Exception:
            return

        diagnostics = output.get("diagnostics", [])
        for diag in diagnostics:
            # Check if this diagnostic is for our function
            if func_name not in diag.get("message", ""):
                continue

            severity = diag.get("severity", "error")
            if severity != "error":
                continue

            diag_line = diag.get("line", 0)
            call_line = diag.get("callLine")

            # Check if this error is for our specific call
            # If callLine is present, use exact matching (the analyzer tracked which call this error is for)
            # Otherwise fall back to range-based heuristic for backward compatibility
            if call_line is not None:
                if call_line != lineno:
                    continue
            else:
                # Fallback: Only report errors that are near the call site
                if not (lineno - 2 <= diag_line <= lineno + 10):
                    continue

            message = diag.get("message", "")
            file = diag.get("file", filename)
            line = diag_line
            code = diag.get("code", "")

            # Extract missing exceptions from suggestions
            missing_names = []
            for suggestion in diag.get("suggestions", []):
                if suggestion.get("action") == "add_handler":
                    missing_names.append(suggestion.get("exception", ""))

            # Build a helpful error message
            error_msg = f"{file}:{line}: {message}"

            # Different help messages based on error type
            if code in ("EXH007", "EXH008"):
                # Unhandled Result/Option
                if signature_type == "option":
                    error_msg += "\n\n  To fix, handle the Option with match:"
                    error_msg += f"\n    match result:"
                    error_msg += f"\n        case Some(value):"
                    error_msg += f"\n            # handle value"
                    error_msg += f"\n        case Nothing():"
                    error_msg += f"\n            # handle nothing"
                else:
                    error_msg += "\n\n  To fix, handle the Result with match:"
                    error_msg += f"\n    match result:"
                    error_msg += f"\n        case Ok(value):"
                    error_msg += f"\n            # handle success"
                    error_msg += f"\n        case Err(e):"
                    error_msg += f"\n            # handle each error type"
            elif missing_names:
                error_msg += f"\n\n  Missing handlers: {', '.join(missing_names)}"
                error_msg += "\n\n  To fix, add the missing case(s) to your match statement:"
                for name in missing_names:
                    if signature_type == "option":
                        if name == "Some":
                            error_msg += f"\n    case Some(value):"
                            error_msg += f"\n        # handle value"
                        else:
                            error_msg += f"\n    case Nothing():"
                            error_msg += f"\n        # handle nothing"
                    else:
                        error_msg += f"\n    case Err({name}() as e):"
                        error_msg += f"\n        # handle {name}"

            raise ExhaustivenessError(
                error_msg,
                func_name=func_name,
                missing=[],
                declared=list(exc_types) if exc_types else [],
            )
    finally:
        del frame


def raises(
    *exc_types: type[BaseException],
) -> Callable[[Callable[P, R]], Callable[P, Result[R, BaseException]]]:
    signature = ExhaustiveSignature(*exc_types)

    def decorator(fn: Callable[P, R]) -> Callable[P, Result[R, BaseException]]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[R, BaseException]:
            # Run static analysis on the caller's code
            _check_caller_exhaustiveness(fn.__name__, exc_types)

            try:
                result = fn(*args, **kwargs)
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


def returns_option(
    fn: Callable[P, R],
) -> Callable[P, R]:
    """
    Decorator that marks a function as returning an Option type.

    This enables exhaustive checking with match() - the caller must handle
    both Some and Nothing cases.

    Usage:
        @returns_option
        def find_user(user_id: str) -> Option[User]:
            user = db.get(user_id)
            if user is None:
                return Nothing()
            return Some(user)

        # This will fail if Nothing case is not handled
        result = match(find_user, "123")({
            Some: lambda user: user.name,
            Nothing: lambda: "Unknown",
        })
    """
    from pyrethrin.option import Nothing, Some

    signature = OptionSignature()

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Run static analysis on the caller's code
        _check_caller_exhaustiveness(fn.__name__, None, "option")

        result = fn(*args, **kwargs)
        # Validate that the function actually returns an Option
        if not isinstance(result, (Some, Nothing)):
            raise TypeError(
                f"{fn.__name__} is decorated with @returns_option but returned "
                f"{type(result).__name__} instead of Some or Nothing"
            )
        return result

    wrapper.__pyrethrin_option_signature__ = signature  # type: ignore[attr-defined]
    return wrapper
