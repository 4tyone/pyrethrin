import pytest

from pyrethrin import Err, Ok


class TestOk:
    def test_value_accessible(self):
        result = Ok(42)
        assert result.value == 42

    def test_is_ok_returns_true(self):
        result = Ok("hello")
        assert result.is_ok() is True

    def test_is_err_returns_false(self):
        result = Ok("hello")
        assert result.is_err() is False

    def test_pattern_matching(self):
        result: Ok[int] | Err[Exception] = Ok(42)
        match result:
            case Ok(value):
                assert value == 42
            case Err(_):
                pytest.fail("Should not match Err")


class TestErr:
    def test_error_accessible(self):
        error = ValueError("test error")
        result = Err(error)
        assert result.error is error

    def test_is_ok_returns_false(self):
        result = Err(ValueError("error"))
        assert result.is_ok() is False

    def test_is_err_returns_true(self):
        result = Err(ValueError("error"))
        assert result.is_err() is True

    def test_pattern_matching(self):
        error = ValueError("test")
        result: Ok[int] | Err[ValueError] = Err(error)
        match result:
            case Ok(_):
                pytest.fail("Should not match Ok")
            case Err(e):
                assert e is error
