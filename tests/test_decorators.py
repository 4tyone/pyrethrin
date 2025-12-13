import pytest

from pyrethrin import Err, Ok, UndeclaredExceptionError, raises


class UserNotFound(Exception):
    pass


class InvalidUserId(Exception):
    pass


class TestRaisesDecorator:
    def test_returns_ok_on_success(self):
        @raises(ValueError)
        def greet(name: str) -> str:
            return f"Hello, {name}"

        result = greet("world")
        assert isinstance(result, Ok)
        assert result.value == "Hello, world"

    def test_returns_err_on_declared_exception(self):
        @raises(ValueError)
        def validate(x: int) -> int:
            if x < 0:
                raise ValueError("must be positive")
            return x

        result = validate(-1)
        assert isinstance(result, Err)
        assert isinstance(result.error, ValueError)

    def test_raises_undeclared_exception_error(self):
        @raises(ValueError)
        def buggy() -> str:
            raise KeyError("oops")

        with pytest.raises(UndeclaredExceptionError) as exc_info:
            buggy()

        assert exc_info.value.fn == "buggy"
        assert exc_info.value.got == "KeyError"
        assert "ValueError" in exc_info.value.declared

    def test_multiple_exception_types(self):
        @raises(UserNotFound, InvalidUserId)
        def get_user(user_id: str) -> dict:
            if not user_id.isalnum():
                raise InvalidUserId(f"Invalid: {user_id}")
            if user_id == "unknown":
                raise UserNotFound(user_id)
            return {"id": user_id}

        assert get_user("abc123").is_ok()

        result = get_user("!!!")
        assert result.is_err()
        assert isinstance(result.error, InvalidUserId)

        result = get_user("unknown")
        assert result.is_err()
        assert isinstance(result.error, UserNotFound)

    def test_preserves_function_name(self):
        @raises(ValueError)
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_attaches_signature_metadata(self):
        @raises(ValueError, TypeError)
        def func() -> None:
            pass

        assert hasattr(func, "__pyrethrin_signature__")
        assert hasattr(func, "__pyrethrin_raises__")
        assert func.__pyrethrin_raises__ == frozenset({ValueError, TypeError})

    def test_passthrough_existing_result(self):
        @raises(ValueError)
        def returns_result() -> Ok[int]:
            return Ok(42)

        result = returns_result()
        assert isinstance(result, Ok)
        assert result.value == 42
