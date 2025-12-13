from __future__ import annotations


class PyrethrinError(Exception):
    pass


def _type_name(t: type) -> str:
    """Get a readable name for a type."""
    module = getattr(t, "__module__", "")
    name = getattr(t, "__name__", str(t))
    if module and module != "builtins" and module != "__main__":
        return f"{module}.{name}"
    return name


def _format_type_list(types: list[type]) -> str:
    """Format a list of types as a readable string."""
    if not types:
        return "(none)"
    names = sorted(_type_name(t) for t in types)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


class ExhaustivenessError(PyrethrinError):
    def __init__(
        self,
        message: str,
        *,
        func_name: str | None = None,
        missing: list[type] | None = None,
        extra: list[type] | None = None,
        provided: list[type] | None = None,
        declared: list[type] | None = None,
    ):
        self.func_name = func_name
        self.missing = missing or []
        self.extra = extra or []
        self.provided = provided or []
        self.declared = declared or []

        full_message = self._build_message(message)
        super().__init__(full_message)

    def _build_message(self, base_message: str) -> str:
        lines = [base_message]

        if self.func_name:
            lines[0] = f"{base_message} for `{self.func_name}`"

        if self.missing:
            lines.append("")
            lines.append(f"  Missing handlers: {_format_type_list(self.missing)}")

        if self.extra:
            lines.append("")
            lines.append(f"  Unexpected handlers: {_format_type_list(self.extra)}")

        if self.declared:
            lines.append("")
            lines.append(f"  Declared exceptions: {_format_type_list(self.declared)}")

        if self.provided:
            provided_names = _format_type_list(self.provided)
            lines.append(f"  Provided handlers: {provided_names}")

        if self.missing:
            lines.append("")
            lines.append("  To fix, add handlers for the missing exceptions:")
            for exc in sorted(self.missing, key=lambda t: _type_name(t)):
                name = _type_name(exc)
                lines.append(f"    {name}: lambda e: ...,")

        if self.extra:
            lines.append("")
            lines.append("  To fix, either:")
            lines.append("    1. Remove the unexpected handlers, or")
            lines.append("    2. Add the exceptions to the @raises decorator")

        return "\n".join(lines)


class UndeclaredExceptionError(PyrethrinError):
    def __init__(
        self,
        fn: str,
        got: str,
        declared: list[str],
        original: BaseException,
    ):
        self.fn = fn
        self.got = got
        self.declared = declared
        self.original = original

        message = self._build_message()
        super().__init__(message)

    def _build_message(self) -> str:
        lines = [
            f"`{self.fn}` raised `{self.got}` which is not in its @raises declaration",
            "",
            f"  Raised: {self.got}",
            f"  Declared: {', '.join(sorted(self.declared)) if self.declared else '(none)'}",
            "",
            "  This is a bug in the function implementation.",
            "  To fix, either:",
            f"    1. Add `{self.got}` to the @raises decorator",
            f"    2. Handle `{self.got}` inside the function before it escapes",
        ]
        return "\n".join(lines)
