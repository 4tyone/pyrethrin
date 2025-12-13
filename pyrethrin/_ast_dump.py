from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Location:
    file: str
    line: int
    col: int
    end_line: int
    end_col: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }


@dataclass
class ExcType:
    kind: str
    name: str | None = None
    module: str | None = None
    types: list[ExcType] | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "name":
            return {"kind": "name", "name": self.name}
        elif self.kind == "qualified":
            return {"kind": "qualified", "module": self.module, "name": self.name}
        elif self.kind == "union":
            return {"kind": "union", "types": [t.to_dict() for t in (self.types or [])]}
        elif self.kind == "ok":
            return {"kind": "ok"}
        elif self.kind == "some":
            return {"kind": "some"}
        elif self.kind == "nothing":
            return {"kind": "nothing"}
        return {"kind": self.kind}


@dataclass
class FuncSignature:
    name: str
    qualified_name: str | None
    declared_exceptions: list[ExcType]
    loc: Location
    is_async: bool
    signature_type: str = "raises"  # "raises" or "option"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "declared_exceptions": [e.to_dict() for e in self.declared_exceptions],
            "loc": self.loc.to_dict(),
            "is_async": self.is_async,
            "signature_type": self.signature_type,
        }


@dataclass
class MatchCall:
    func_name: str
    handlers: list[ExcType]
    has_ok_handler: bool
    loc: Location
    kind: str
    has_some_handler: bool = False
    has_nothing_handler: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "func_name": self.func_name,
            "handlers": [h.to_dict() for h in self.handlers],
            "has_ok_handler": self.has_ok_handler,
            "has_some_handler": self.has_some_handler,
            "has_nothing_handler": self.has_nothing_handler,
            "loc": self.loc.to_dict(),
            "kind": self.kind,
        }


@dataclass
class UnhandledCall:
    """A call to a @raises or @returns_option function that is not handled with match."""

    func_name: str
    loc: Location
    signature_type: str  # "raises" or "option"

    def to_dict(self) -> dict[str, Any]:
        return {
            "func_name": self.func_name,
            "loc": self.loc.to_dict(),
            "signature_type": self.signature_type,
        }


@dataclass
class AnalysisResult:
    signatures: list[FuncSignature] = field(default_factory=list)
    matches: list[MatchCall] = field(default_factory=list)
    unhandled_calls: list[UnhandledCall] = field(default_factory=list)
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "signatures": [s.to_dict() for s in self.signatures],
            "matches": [m.to_dict() for m in self.matches],
            "unhandled_calls": [u.to_dict() for u in self.unhandled_calls],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _get_loc(node: ast.AST, file_path: str) -> Location:
    return Location(
        file=file_path,
        line=node.lineno,
        col=node.col_offset,
        end_line=node.end_lineno or node.lineno,
        end_col=node.end_col_offset or node.col_offset,
    )


def _extract_exception_from_decorator_arg(arg: ast.expr) -> ExcType | None:
    if isinstance(arg, ast.Name):
        return ExcType(kind="name", name=arg.id)
    elif isinstance(arg, ast.Attribute):
        parts = []
        node: ast.expr = arg
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            parts.reverse()
            if len(parts) == 2:
                return ExcType(kind="qualified", module=parts[0], name=parts[1])
            return ExcType(kind="qualified", module=".".join(parts[:-1]), name=parts[-1])
    return None


def _extract_handler_from_match_key(key: ast.expr) -> ExcType | None:
    if isinstance(key, ast.Name):
        if key.id == "Ok":
            return ExcType(kind="ok")
        if key.id == "Some":
            return ExcType(kind="some")
        if key.id == "Nothing":
            return ExcType(kind="nothing")
        return ExcType(kind="name", name=key.id)
    elif isinstance(key, ast.Attribute):
        parts = []
        node: ast.expr = key
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            parts.reverse()
            if parts[-1] == "Ok":
                return ExcType(kind="ok")
            if parts[-1] == "Some":
                return ExcType(kind="some")
            if parts[-1] == "Nothing":
                return ExcType(kind="nothing")
            if len(parts) == 2:
                return ExcType(kind="qualified", module=parts[0], name=parts[1])
            return ExcType(kind="qualified", module=".".join(parts[:-1]), name=parts[-1])
    return None


class RaisesVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.signatures: list[FuncSignature] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node, is_async=True)
        self.generic_visit(node)

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        for decorator in node.decorator_list:
            # Check for @raises / @async_raises
            exc_types = self._extract_raises_decorator(decorator)
            if exc_types is not None:
                qualified = None
                if self._class_stack:
                    qualified = ".".join(self._class_stack) + "." + node.name
                self.signatures.append(
                    FuncSignature(
                        name=node.name,
                        qualified_name=qualified,
                        declared_exceptions=exc_types,
                        loc=_get_loc(node, self.file_path),
                        is_async=is_async,
                        signature_type="raises",
                    )
                )
                break

            # Check for @returns_option
            if self._is_returns_option_decorator(decorator):
                qualified = None
                if self._class_stack:
                    qualified = ".".join(self._class_stack) + "." + node.name
                self.signatures.append(
                    FuncSignature(
                        name=node.name,
                        qualified_name=qualified,
                        declared_exceptions=[],  # Option doesn't have declared exceptions
                        loc=_get_loc(node, self.file_path),
                        is_async=is_async,
                        signature_type="option",
                    )
                )
                break

    def _is_returns_option_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is @returns_option."""
        if isinstance(decorator, ast.Name):
            return decorator.id == "returns_option"
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr == "returns_option"
        return False

    def _extract_raises_decorator(self, decorator: ast.expr) -> list[ExcType] | None:
        if isinstance(decorator, ast.Call):
            func = decorator.func
            is_raises = (isinstance(func, ast.Name) and func.id in ("raises", "async_raises")) or (
                isinstance(func, ast.Attribute) and func.attr in ("raises", "async_raises")
            )
            if is_raises:
                return self._extract_decorator_args(decorator)
        return None

    def _extract_decorator_args(self, call: ast.Call) -> list[ExcType]:
        exc_types = []
        for arg in call.args:
            exc_type = _extract_exception_from_decorator_arg(arg)
            if exc_type:
                exc_types.append(exc_type)
        return exc_types


@dataclass
class ResultBinding:
    var_name: str
    func_name: str
    loc: Location
    call_index: int  # Index into _function_calls to link binding to specific call


@dataclass
class FunctionCall:
    """Tracks a call to a function that might need handling."""

    func_name: str
    loc: Location
    var_name: str | None = None  # If assigned to a variable
    handled: bool = False  # Whether this call was handled by match


class MatchVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, known_signatures: set[str] | None = None):
        self.file_path = file_path
        self.matches: list[MatchCall] = []
        self._result_bindings: dict[str, ResultBinding] = {}  # Current binding per var name
        self._function_calls: list[FunctionCall] = []
        self._known_signatures = known_signatures or set()

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            func_name = self._extract_func_call_name(node.value)
            if func_name:
                # Track as a potential unhandled call
                call_index = -1
                if func_name in self._known_signatures:
                    call_index = len(self._function_calls)
                    self._function_calls.append(
                        FunctionCall(
                            func_name=func_name,
                            loc=_get_loc(node.value, self.file_path),
                            var_name=var_name,
                        )
                    )
                # Update binding - this REPLACES any previous binding for this var
                self._result_bindings[var_name] = ResultBinding(
                    var_name=var_name,
                    func_name=func_name,
                    loc=_get_loc(node, self.file_path),
                    call_index=call_index,
                )
        self.generic_visit(node)

    def _extract_func_call_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return node.func.attr
        return None

    def visit_Call(self, node: ast.Call) -> None:
        self._check_match_function_call(node)
        self.generic_visit(node)

    def _check_match_function_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Call):
            return
        inner_call = node.func
        if not (isinstance(inner_call.func, ast.Name) and inner_call.func.id == "match"):
            if isinstance(inner_call.func, ast.Attribute):
                if inner_call.func.attr != "match":
                    return
            else:
                return

        func_name = self._extract_match_func_target(inner_call)
        if not func_name:
            return

        # Mark this function as handled via match()
        # The match() function directly calls the function, so we need to mark it
        # by function name (not var name)
        self._mark_func_handled_by_name(func_name)

        handlers, has_ok, has_some, has_nothing = self._extract_handlers_from_dict(node.args)
        self.matches.append(
            MatchCall(
                func_name=func_name,
                handlers=handlers,
                has_ok_handler=has_ok,
                has_some_handler=has_some,
                has_nothing_handler=has_nothing,
                loc=_get_loc(node, self.file_path),
                kind="function_call",
            )
        )

    def _mark_func_handled_by_name(self, func_name: str) -> None:
        """Mark all calls to this function as handled (used for match() which calls directly)."""
        # For match(func, args), the function is called inside match, so we don't
        # track it separately - it's handled by the match() call itself
        pass

    def _extract_match_func_target(self, call: ast.Call) -> str | None:
        if call.args:
            first_arg = call.args[0]
            if isinstance(first_arg, ast.Name):
                return first_arg.id
            elif isinstance(first_arg, ast.Attribute):
                return first_arg.attr
        return None

    def _extract_handlers_from_dict(
        self, args: list[ast.expr]
    ) -> tuple[list[ExcType], bool, bool, bool]:
        handlers = []
        has_ok = False
        has_some = False
        has_nothing = False
        if args and isinstance(args[0], ast.Dict):
            dict_node = args[0]
            for key in dict_node.keys:
                if key is None:
                    continue
                handler = _extract_handler_from_match_key(key)
                if handler:
                    if handler.kind == "ok":
                        has_ok = True
                    elif handler.kind == "some":
                        has_some = True
                    elif handler.kind == "nothing":
                        has_nothing = True
                    else:
                        handlers.append(handler)
        return handlers, has_ok, has_some, has_nothing

    def visit_Match(self, node: ast.Match) -> None:
        subject = node.subject
        func_name = None
        call_index = -1

        if isinstance(subject, ast.Name):
            var_name = subject.id
            binding = self._result_bindings.get(var_name)
            if binding:
                func_name = binding.func_name
                call_index = binding.call_index

        if func_name:
            # Mark the SPECIFIC function call as handled (by index)
            if call_index >= 0 and call_index < len(self._function_calls):
                self._function_calls[call_index].handled = True

            handlers, has_ok, has_some, has_nothing = self._extract_handlers_from_match_cases(
                node.cases
            )
            self.matches.append(
                MatchCall(
                    func_name=func_name,
                    handlers=handlers,
                    has_ok_handler=has_ok,
                    has_some_handler=has_some,
                    has_nothing_handler=has_nothing,
                    loc=_get_loc(node, self.file_path),
                    kind="statement",
                )
            )

        self.generic_visit(node)

    def get_unhandled_calls(self) -> list[FunctionCall]:
        """Returns list of function calls that were not handled by match."""
        return [call for call in self._function_calls if not call.handled]

    def _extract_handlers_from_match_cases(
        self, cases: list[ast.match_case]
    ) -> tuple[list[ExcType], bool, bool, bool]:
        handlers = []
        has_ok = False
        has_some = False
        has_nothing = False
        for case in cases:
            pattern = case.pattern
            handler, is_ok, is_some, is_nothing = self._extract_handler_from_pattern(pattern)
            if handler:
                if is_ok:
                    has_ok = True
                elif is_some:
                    has_some = True
                elif is_nothing:
                    has_nothing = True
                else:
                    handlers.append(handler)
            elif is_ok:
                has_ok = True
            elif is_some:
                has_some = True
            elif is_nothing:
                has_nothing = True
        return handlers, has_ok, has_some, has_nothing

    def _extract_handler_from_pattern(
        self, pattern: ast.pattern
    ) -> tuple[ExcType | None, bool, bool, bool]:
        """Returns (handler, is_ok, is_some, is_nothing)."""
        if isinstance(pattern, ast.MatchClass):
            cls = pattern.cls
            if isinstance(cls, ast.Name):
                if cls.id == "Ok":
                    return None, True, False, False
                if cls.id == "Some":
                    return None, False, True, False
                if cls.id == "Nothing":
                    return None, False, False, True
                if cls.id == "Err" and pattern.patterns:
                    inner = pattern.patterns[0]
                    exc, _ = self._extract_exc_from_err_pattern(inner)
                    return exc, False, False, False
                return ExcType(kind="name", name=cls.id), False, False, False
            elif isinstance(cls, ast.Attribute):
                parts = self._collect_attribute_parts(cls)
                if parts[-1] == "Ok":
                    return None, True, False, False
                if parts[-1] == "Some":
                    return None, False, True, False
                if parts[-1] == "Nothing":
                    return None, False, False, True
                if parts[-1] == "Err" and pattern.patterns:
                    inner = pattern.patterns[0]
                    exc, _ = self._extract_exc_from_err_pattern(inner)
                    return exc, False, False, False
                if len(parts) >= 2:
                    return (
                        ExcType(
                            kind="qualified",
                            module=".".join(parts[:-1]),
                            name=parts[-1],
                        ),
                        False,
                        False,
                        False,
                    )
        return None, False, False, False

    def _extract_exc_from_err_pattern(self, pattern: ast.pattern) -> tuple[ExcType | None, bool]:
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                return self._extract_exc_from_err_pattern(pattern.pattern)
            return None, False
        if isinstance(pattern, ast.MatchClass):
            cls = pattern.cls
            if isinstance(cls, ast.Name):
                return ExcType(kind="name", name=cls.id), False
            elif isinstance(cls, ast.Attribute):
                parts = self._collect_attribute_parts(cls)
                if len(parts) >= 2:
                    return (
                        ExcType(
                            kind="qualified",
                            module=".".join(parts[:-1]),
                            name=parts[-1],
                        ),
                        False,
                    )
        return None, False

    def _collect_attribute_parts(self, node: ast.Attribute) -> list[str]:
        parts = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        return parts


def analyze_file(file_path: str | Path) -> AnalysisResult:
    file_path = Path(file_path)
    source = file_path.read_text()
    return analyze_source(source, str(file_path))


def analyze_source(source: str, file_path: str = "<string>") -> AnalysisResult:
    tree = ast.parse(source)

    raises_visitor = RaisesVisitor(file_path)
    raises_visitor.visit(tree)

    # Build a map of known signatures and their types
    signature_map: dict[str, str] = {}  # func_name -> signature_type
    for sig in raises_visitor.signatures:
        signature_map[sig.name] = sig.signature_type

    match_visitor = MatchVisitor(file_path, known_signatures=set(signature_map.keys()))
    match_visitor.visit(tree)

    # Collect unhandled calls
    unhandled_calls = []
    for call in match_visitor.get_unhandled_calls():
        sig_type = signature_map.get(call.func_name, "raises")
        unhandled_calls.append(
            UnhandledCall(
                func_name=call.func_name,
                loc=call.loc,
                signature_type=sig_type,
            )
        )

    return AnalysisResult(
        signatures=raises_visitor.signatures,
        matches=match_visitor.matches,
        unhandled_calls=unhandled_calls,
    )


def dump_file(file_path: str | Path) -> dict[str, Any]:
    result = analyze_file(file_path)
    return result.to_dict()


def dump_file_json(file_path: str | Path, indent: int | None = 2) -> str:
    result = analyze_file(file_path)
    return result.to_json(indent=indent)
