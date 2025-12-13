"""
Edge case tests for static analysis integration.
Tests various patterns that the AST analyzer must handle correctly.
"""

import json

import pytest

from pyrethrin._ast_dump import analyze_source


class TestNestedMatch:
    def test_nested_match_statements(self):
        source = """
@raises(ValueError)
def outer():
    pass

@raises(KeyError)
def inner():
    pass

result1 = outer()
match result1:
    case Ok(v):
        result2 = inner()
        match result2:
            case Ok(v2):
                print(v2)
            case Err(KeyError() as e):
                print(e)
    case Err(ValueError() as e):
        print(e)
"""
        result = analyze_source(source)
        assert len(result.signatures) == 2
        assert len(result.matches) == 2

        outer_match = next(m for m in result.matches if m.func_name == "outer")
        inner_match = next(m for m in result.matches if m.func_name == "inner")

        assert outer_match.has_ok_handler is True
        assert len(outer_match.handlers) == 1
        assert outer_match.handlers[0].name == "ValueError"

        assert inner_match.has_ok_handler is True
        assert len(inner_match.handlers) == 1
        assert inner_match.handlers[0].name == "KeyError"


class TestMatchInControlFlow:
    def test_match_inside_if(self):
        source = """
@raises(ValueError)
def func():
    pass

if condition:
    result = func()
    match result:
        case Ok(v):
            print(v)
        case Err(ValueError() as e):
            print(e)
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert len(result.matches) == 1
        assert result.matches[0].func_name == "func"

    def test_match_inside_for_loop(self):
        source = """
@raises(IOError)
def read_file(path):
    pass

for path in paths:
    result = read_file(path)
    match result:
        case Ok(content):
            process(content)
        case Err(IOError() as e):
            log_error(e)
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        assert result.matches[0].func_name == "read_file"

    def test_match_inside_while_loop(self):
        source = """
@raises(StopIteration)
def get_next():
    pass

while True:
    result = get_next()
    match result:
        case Ok(item):
            yield item
        case Err(StopIteration()):
            break
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        assert result.matches[0].func_name == "get_next"

    def test_match_inside_try_block(self):
        source = """
@raises(ValueError)
def parse(data):
    pass

try:
    result = parse(data)
    match result:
        case Ok(v):
            return v
        case Err(ValueError() as e):
            raise e
except Exception:
    pass
"""
        result = analyze_source(source)
        assert len(result.matches) == 1


class TestQualifiedExceptions:
    def test_module_qualified_exception_in_raises(self):
        source = """
@raises(mymodule.CustomError, pkg.subpkg.AnotherError)
def func():
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert len(sig.declared_exceptions) == 2

        exc1 = sig.declared_exceptions[0]
        assert exc1.kind == "qualified"
        assert exc1.module == "mymodule"
        assert exc1.name == "CustomError"

        exc2 = sig.declared_exceptions[1]
        assert exc2.kind == "qualified"
        assert exc2.module == "pkg.subpkg"
        assert exc2.name == "AnotherError"

    def test_qualified_exception_in_match_handler(self):
        source = """
@raises(errors.NotFoundError)
def find():
    pass

result = find()
match result:
    case Ok(v):
        print(v)
    case Err(errors.NotFoundError() as e):
        print(e)
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.has_ok_handler is True
        assert len(m.handlers) == 1
        h = m.handlers[0]
        assert h.kind == "qualified"
        assert h.module == "errors"
        assert h.name == "NotFoundError"


class TestAsyncFunctions:
    def test_async_raises_decorator(self):
        source = """
@async_raises(ConnectionError, TimeoutError)
async def fetch_data(url):
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert sig.name == "fetch_data"
        assert sig.is_async is True
        assert len(sig.declared_exceptions) == 2

    def test_async_function_match(self):
        source = """
@async_raises(IOError)
async def read_async():
    pass

result = read_async()
match result:
    case Ok(data):
        process(data)
    case Err(IOError() as e):
        handle(e)
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert result.signatures[0].is_async is True
        assert len(result.matches) == 1


class TestClassMethods:
    def test_instance_method(self):
        source = """
class UserService:
    @raises(UserNotFound)
    def get_user(self, user_id):
        pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert sig.name == "get_user"
        assert sig.qualified_name == "UserService.get_user"

    def test_class_method(self):
        source = """
class Factory:
    @classmethod
    @raises(ConfigError)
    def create(cls):
        pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert sig.name == "create"
        assert sig.qualified_name == "Factory.create"

    def test_static_method(self):
        source = """
class Utils:
    @staticmethod
    @raises(ParseError)
    def parse(data):
        pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        sig = result.signatures[0]
        assert sig.name == "parse"
        assert sig.qualified_name == "Utils.parse"

    def test_nested_class_method(self):
        source = """
class Outer:
    class Inner:
        @raises(ValueError)
        def method(self):
            pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 1
        assert result.signatures[0].qualified_name == "Outer.Inner.method"


class TestMultipleFunctions:
    def test_multiple_raises_functions_same_file(self):
        source = """
@raises(ValueError)
def validate_name(name):
    pass

@raises(KeyError)
def get_config(key):
    pass

@raises(IOError, PermissionError)
def save_file(path, content):
    pass

@raises(ConnectionError)
def connect(host):
    pass
"""
        result = analyze_source(source)
        assert len(result.signatures) == 4

        names = {s.name for s in result.signatures}
        assert names == {"validate_name", "get_config", "save_file", "connect"}

        save_file_sig = next(s for s in result.signatures if s.name == "save_file")
        assert len(save_file_sig.declared_exceptions) == 2

    def test_multiple_matches_same_file(self):
        source = """
@raises(ValueError)
def func1():
    pass

@raises(KeyError)
def func2():
    pass

r1 = func1()
match r1:
    case Ok(v):
        pass
    case Err(ValueError()):
        pass

r2 = func2()
match r2:
    case Ok(v):
        pass
    case Err(KeyError()):
        pass
"""
        result = analyze_source(source)
        assert len(result.matches) == 2
        func_names = {m.func_name for m in result.matches}
        assert func_names == {"func1", "func2"}


class TestWildcardPatterns:
    def test_wildcard_after_specific_handlers(self):
        source = """
@raises(ValueError, KeyError, TypeError)
def risky():
    pass

result = risky()
match result:
    case Ok(v):
        print(v)
    case Err(ValueError() as e):
        handle_value_error(e)
    case _:
        handle_other()
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.has_ok_handler is True
        assert len(m.handlers) == 1
        assert m.handlers[0].name == "ValueError"

    def test_wildcard_as_only_error_handler(self):
        source = """
@raises(ValueError)
def func():
    pass

result = func()
match result:
    case Ok(v):
        print(v)
    case _:
        print("error")
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.has_ok_handler is True
        assert len(m.handlers) == 0


class TestMatchFunctionCall:
    def test_match_function_with_dict_handlers(self):
        source = """
@raises(ValueError, KeyError)
def get_data():
    pass

result = match(get_data)(
    {
        Ok: lambda v: v,
        ValueError: lambda e: str(e),
        KeyError: lambda e: "not found",
    }
)
"""
        result = analyze_source(source)
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.kind == "function_call"
        assert m.func_name == "get_data"
        assert m.has_ok_handler is True
        assert len(m.handlers) == 2


class TestComplexScenarios:
    def test_full_realistic_file(self):
        source = """
from pyrethrin import raises, match, Ok, Err
from myapp.errors import UserNotFound, InvalidInput
from myapp.models import User

class UserRepository:
    @raises(UserNotFound, InvalidInput)
    def get_user(self, user_id: str) -> User:
        if not user_id:
            raise InvalidInput("user_id cannot be empty")
        user = self.db.find(user_id)
        if user is None:
            raise UserNotFound(user_id)
        return user

    @raises(InvalidInput)
    def create_user(self, name: str) -> User:
        if not name:
            raise InvalidInput("name cannot be empty")
        return self.db.create(name)

def handle_request(repo: UserRepository, user_id: str):
    result = repo.get_user(user_id)
    match result:
        case Ok(user):
            return {"status": "ok", "user": user.to_dict()}
        case Err(UserNotFound() as e):
            return {"status": "error", "code": 404, "message": str(e)}
        case Err(InvalidInput() as e):
            return {"status": "error", "code": 400, "message": str(e)}
"""
        result = analyze_source(source)

        assert len(result.signatures) == 2
        get_user_sig = next(s for s in result.signatures if s.name == "get_user")
        assert get_user_sig.qualified_name == "UserRepository.get_user"
        assert len(get_user_sig.declared_exceptions) == 2

        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.func_name == "get_user"
        assert m.has_ok_handler is True
        assert len(m.handlers) == 2

    def test_json_output_format_complete(self):
        source = """
@raises(ValueError)
def func():
    pass

result = func()
match result:
    case Ok(v):
        pass
    case Err(ValueError()):
        pass
"""
        result = analyze_source(source, "test_file.py")
        data = result.to_dict()

        assert data["language"] == "python"
        assert len(data["signatures"]) == 1
        assert len(data["matches"]) == 1

        sig = data["signatures"][0]
        assert "name" in sig
        assert "qualified_name" in sig
        assert "declared_exceptions" in sig
        assert "loc" in sig
        assert "is_async" in sig

        loc = sig["loc"]
        assert loc["file"] == "test_file.py"
        assert "line" in loc
        assert "col" in loc
        assert "end_line" in loc
        assert "end_col" in loc
