import json

import pytest

from pyrethrin._ast_dump import (
    AnalysisResult,
    ExcType,
    FuncSignature,
    Location,
    MatchCall,
    analyze_source,
)


class TestLocation:
    def test_to_dict(self):
        loc = Location(file="test.py", line=1, col=0, end_line=1, end_col=10)
        assert loc.to_dict() == {
            "file": "test.py",
            "line": 1,
            "col": 0,
            "end_line": 1,
            "end_col": 10,
        }


class TestExcType:
    def test_name_to_dict(self):
        exc = ExcType(kind="name", name="ValueError")
        assert exc.to_dict() == {"kind": "name", "name": "ValueError"}

    def test_qualified_to_dict(self):
        exc = ExcType(kind="qualified", module="mymodule", name="MyError")
        assert exc.to_dict() == {"kind": "qualified", "module": "mymodule", "name": "MyError"}

    def test_ok_to_dict(self):
        exc = ExcType(kind="ok")
        assert exc.to_dict() == {"kind": "ok"}

    def test_union_to_dict(self):
        exc = ExcType(
            kind="union",
            types=[
                ExcType(kind="name", name="ValueError"),
                ExcType(kind="name", name="KeyError"),
            ],
        )
        assert exc.to_dict() == {
            "kind": "union",
            "types": [
                {"kind": "name", "name": "ValueError"},
                {"kind": "name", "name": "KeyError"},
            ],
        }


class TestRaisesVisitor:
    def test_find_simple_raises_decorator(self):
        source = """
from pyrethrin import raises

@raises(ValueError)
def foo():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert sig.name == "foo"
        assert sig.is_async is False
        assert len(sig.declared_exceptions) == 1
        assert sig.declared_exceptions[0].kind == "name"
        assert sig.declared_exceptions[0].name == "ValueError"

    def test_find_multiple_exceptions(self):
        source = """
@raises(ValueError, KeyError, TypeError)
def bar():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert len(result.signatures[0].declared_exceptions) == 3
        names = [e.name for e in result.signatures[0].declared_exceptions]
        assert names == ["ValueError", "KeyError", "TypeError"]

    def test_find_async_raises(self):
        source = """
@async_raises(IOError)
async def fetch():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert result.signatures[0].is_async is True
        assert result.signatures[0].name == "fetch"

    def test_find_qualified_exception(self):
        source = """
@raises(mymodule.MyError)
def baz():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        exc = result.signatures[0].declared_exceptions[0]
        assert exc.kind == "qualified"
        assert exc.module == "mymodule"
        assert exc.name == "MyError"

    def test_find_deeply_qualified_exception(self):
        source = """
@raises(pkg.subpkg.errors.CustomError)
def deep():
    pass
"""
        result = analyze_source(source)
        exc = result.signatures[0].declared_exceptions[0]
        assert exc.kind == "qualified"
        assert exc.module == "pkg.subpkg.errors"
        assert exc.name == "CustomError"

    def test_class_method_qualified_name(self):
        source = """
class MyClass:
    @raises(ValueError)
    def method(self):
        pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert result.signatures[0].name == "method"
        assert result.signatures[0].qualified_name == "MyClass.method"

    def test_nested_class_qualified_name(self):
        source = """
class Outer:
    class Inner:
        @raises(ValueError)
        def nested(self):
            pass
"""
        result = analyze_source(source)
        assert result.signatures[0].qualified_name == "Outer.Inner.nested"

    def test_no_raises_decorator(self):
        source = """
def plain_function():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 0


class TestMatchVisitor:
    def test_find_match_function_call(self):
        source = """
result = match(foo)(
    {
        Ok: lambda v: v,
        ValueError: lambda e: str(e),
    }
)
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.func_name == "foo"
        assert m.kind == "function_call"
        assert m.has_ok_handler is True
        assert len(m.handlers) == 1
        assert m.handlers[0].name == "ValueError"

    def test_find_match_statement(self):
        source = """
result = fetch_data()
match result:
    case Ok(value):
        print(value)
    case Err(ValueError() as e):
        print(e)
    case Err(IOError() as e):
        log(e)
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.func_name == "fetch_data"
        assert m.kind == "statement"
        assert m.has_ok_handler is True
        assert len(m.handlers) == 2
        handler_names = {h.name for h in m.handlers}
        assert handler_names == {"ValueError", "IOError"}

    def test_result_variable_tracking(self):
        source = """
x = some_func()
y = another_func()
match x:
    case Ok(v):
        pass
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        assert result.matches[0].func_name == "some_func"

    def test_match_with_qualified_exception_handler(self):
        source = """
r = process()
match r:
    case Ok(v):
        pass
    case Err(custom.errors.MyError() as e):
        pass
"""
        result = analyze_source(source)
        m = result.matches[0]
        assert m.has_ok_handler is True
        assert len(m.handlers) == 1
        h = m.handlers[0]
        assert h.kind == "qualified"
        assert h.module == "custom.errors"
        assert h.name == "MyError"

    def test_untracked_variable_not_analyzed(self):
        source = """
match unknown_var:
    case Ok(v):
        pass
"""
        result = analyze_source(source)
        assert len(result.matches) == 0


class TestAnalysisResult:
    def test_to_json(self):
        result = AnalysisResult(
            signatures=[
                FuncSignature(
                    name="test",
                    qualified_name=None,
                    declared_exceptions=[ExcType(kind="name", name="ValueError")],
                    loc=Location("test.py", 1, 0, 3, 10),
                    is_async=False,
                )
            ],
            matches=[
                MatchCall(
                    func_name="test",
                    handlers=[ExcType(kind="name", name="ValueError")],
                    has_ok_handler=True,
                    loc=Location("test.py", 5, 0, 10, 1),
                    kind="function_call",
                )
            ],
        )
        data = json.loads(result.to_json())
        assert len(data["signatures"]) == 1
        assert len(data["matches"]) == 1
        assert data["signatures"][0]["name"] == "test"
        assert data["matches"][0]["func_name"] == "test"


class TestIntegration:
    def test_full_file_analysis(self):
        source = """
from pyrethrin import raises, match, Ok, Err

@raises(ValueError, KeyError)
def get_data(key: str) -> str:
    if not key:
        raise ValueError("empty key")
    return "value"

result = get_data("test")
match result:
    case Ok(value):
        print(value)
    case Err(ValueError() as e):
        print(f"Value error: {e}")
    case Err(KeyError() as e):
        print(f"Key error: {e}")
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert len(result.matches) == 1

        sig = result.signatures[0]
        assert sig.name == "get_data"
        assert len(sig.declared_exceptions) == 2

        m = result.matches[0]
        assert m.func_name == "get_data"
        assert m.has_ok_handler is True
        assert len(m.handlers) == 2

    def test_pyrethrum_json_format(self):
        source = """
@raises(ValueError)
def foo():
    pass

r = foo()
match r:
    case Ok(v):
        pass
    case Err(ValueError()):
        pass
"""
        result = analyze_source(source, "example.py")
        data = result.to_dict()

        assert "signatures" in data
        assert "matches" in data

        sig = data["signatures"][0]
        assert sig["name"] == "foo"
        assert sig["qualified_name"] is None
        assert sig["is_async"] is False
        assert sig["loc"]["file"] == "example.py"
        assert len(sig["declared_exceptions"]) == 1
        assert sig["declared_exceptions"][0] == {"kind": "name", "name": "ValueError"}

        match = data["matches"][0]
        assert match["func_name"] == "foo"
        assert match["kind"] == "statement"
        assert match["has_ok_handler"] is True
        assert len(match["handlers"]) == 1
