"""Tests for combining @raises and @returns_option decorators."""

import pytest

from pyrethrin import Err, Ok, raises
from pyrethrin.decorators import returns_option
from pyrethrin.option import Nothing, Option, Some


class NetworkError(Exception):
    pass


class TestCombinedDecorators:
    """Test what happens when both @raises and @returns_option are applied."""

    def test_raises_wrapping_returns_option(self):
        """
        @raises on top of @returns_option.

        Expected behavior: The @returns_option function returns Some/Nothing,
        then @raises wraps that in Ok (since no exception was raised).
        Result type: Result[Option[T], E]
        """

        @raises(NetworkError)
        @returns_option
        def find_user(user_id: str) -> Option[str]:
            if user_id == "error":
                raise NetworkError("Connection failed")
            if user_id == "none":
                return Nothing()
            return Some(f"User-{user_id}")

        # Success case - returns Ok(Some(...))
        result = find_user("123")
        assert isinstance(result, Ok)
        assert isinstance(result.value, Some)
        assert result.value.value == "User-123"

        # Nothing case - returns Ok(Nothing())
        result = find_user("none")
        assert isinstance(result, Ok)
        assert isinstance(result.value, Nothing)

        # Exception case - returns Err(NetworkError)
        result = find_user("error")
        assert isinstance(result, Err)
        assert isinstance(result.error, NetworkError)

    def test_returns_option_wrapping_raises_fails(self):
        """
        @returns_option on top of @raises.

        Expected behavior: @raises returns Result (Ok/Err), but @returns_option
        expects Option (Some/Nothing), so it should raise TypeError.
        """

        @returns_option
        @raises(ValueError)
        def compute(x: int) -> int:
            if x < 0:
                raise ValueError("Must be positive")
            return x * 2

        # This should fail because @raises returns Ok/Err, not Some/Nothing
        with pytest.raises(TypeError) as exc_info:
            compute(5)

        assert "returned" in str(exc_info.value)
        assert "Some or Nothing" in str(exc_info.value)

    def test_pattern_matching_nested_result_option(self):
        """Test exhaustive pattern matching on Result[Option[T], E]."""

        @raises(NetworkError)
        @returns_option
        def fetch_data(key: str) -> Option[int]:
            if key == "error":
                raise NetworkError("Failed")
            if key == "missing":
                return Nothing()
            return Some(42)

        # Match on the outer Result first
        result = fetch_data("valid")
        match result:
            case Ok(inner):
                # Then match on the inner Option
                match inner:
                    case Some(value):
                        assert value == 42
                    case Nothing():
                        pytest.fail("Expected Some")
            case Err(e):
                pytest.fail(f"Expected Ok, got Err: {e}")

        # Test Nothing case
        result = fetch_data("missing")
        match result:
            case Ok(inner):
                match inner:
                    case Some(_):
                        pytest.fail("Expected Nothing")
                    case Nothing():
                        pass  # Expected
            case Err(_):
                pytest.fail("Expected Ok")

        # Test Error case
        result = fetch_data("error")
        match result:
            case Ok(_):
                pytest.fail("Expected Err")
            case Err(e):
                assert isinstance(e, NetworkError)

    def test_metadata_preserved_with_both_decorators(self):
        """Check which metadata is preserved when both decorators are applied."""

        @raises(ValueError)
        @returns_option
        def my_func() -> Option[str]:
            return Some("test")

        # The outer decorator (@raises) should set its metadata
        assert hasattr(my_func, "__pyrethrin_signature__")
        assert hasattr(my_func, "__pyrethrin_raises__")
        assert ValueError in my_func.__pyrethrin_raises__

        # The inner @returns_option metadata may be lost due to wrapping
        # This is expected behavior - the outer decorator wins

    def test_exception_from_returns_option_validation(self):
        """
        Test when @returns_option itself raises a TypeError.

        If the inner function doesn't return Some/Nothing,
        @returns_option raises TypeError, which @raises won't catch
        unless TypeError is declared.
        """

        @raises(ValueError)
        @returns_option
        def bad_return() -> Option[str]:
            return "not an option"  # type: ignore - intentionally wrong

        # TypeError from @returns_option is not in @raises declaration
        # so it should propagate as UndeclaredExceptionError
        from pyrethrin import UndeclaredExceptionError

        with pytest.raises(UndeclaredExceptionError) as exc_info:
            bad_return()

        assert "TypeError" in exc_info.value.got

    def test_raises_catches_option_validation_error_when_declared(self):
        """When TypeError is declared, @raises will catch the validation error."""

        @raises(TypeError)
        @returns_option
        def bad_return() -> Option[str]:
            return "not an option"  # type: ignore - intentionally wrong

        result = bad_return()
        assert isinstance(result, Err)
        assert isinstance(result.error, TypeError)
