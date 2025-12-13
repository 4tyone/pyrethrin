import pytest

from pyrethrin import (
    Err,
    ExhaustivenessError,
    Nothing,
    Ok,
    Option,
    Some,
    match,
    raises,
    returns_option,
)


class NotFound(Exception):
    pass


class InvalidInput(Exception):
    pass


@raises(NotFound, InvalidInput)
def fetch_item(item_id: str) -> dict:
    if not item_id:
        raise InvalidInput("empty id")
    if item_id == "missing":
        raise NotFound(item_id)
    return {"id": item_id, "name": "Item"}


class TestMatch:
    def test_dispatches_to_ok_handler(self):
        result = match(fetch_item, "abc123")(
            {
                Ok: lambda item: f"Found: {item['name']}",
                NotFound: lambda e: "not found",
                InvalidInput: lambda e: "invalid",
            }
        )
        assert result == "Found: Item"

    def test_dispatches_to_error_handler(self):
        result = match(fetch_item, "missing")(
            {
                Ok: lambda item: "found",
                NotFound: lambda e: f"Not found: {e}",
                InvalidInput: lambda e: "invalid",
            }
        )
        assert "Not found" in result

    def test_raises_on_missing_ok_handler(self):
        with pytest.raises(ExhaustivenessError) as exc_info:
            match(fetch_item, "abc")(
                {
                    NotFound: lambda e: "not found",
                    InvalidInput: lambda e: "invalid",
                }
            )
        assert Ok in exc_info.value.missing

    def test_raises_on_missing_error_handler(self):
        with pytest.raises(ExhaustivenessError) as exc_info:
            match(fetch_item, "abc")(
                {
                    Ok: lambda item: "found",
                    NotFound: lambda e: "not found",
                }
            )
        assert InvalidInput in exc_info.value.missing

    def test_raises_on_extra_handler(self):
        class UnrelatedError(Exception):
            pass

        with pytest.raises(ExhaustivenessError) as exc_info:
            match(fetch_item, "abc")(
                {
                    Ok: lambda item: "found",
                    NotFound: lambda e: "not found",
                    InvalidInput: lambda e: "invalid",
                    UnrelatedError: lambda e: "unrelated",
                }
            )
        assert UnrelatedError in exc_info.value.extra

    def test_requires_raises_decorated_function(self):
        def not_decorated(x: int) -> int:
            return x

        with pytest.raises(
            TypeError, match=r"requires a @raises or @returns_option decorated function"
        ):
            match(not_decorated, 42)

    def test_validates_before_execution(self):
        call_count = 0

        @raises(ValueError)
        def side_effect() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        with pytest.raises(ExhaustivenessError):
            match(side_effect)(
                {
                    Ok: lambda x: x,
                }
            )

        assert call_count == 0

    def test_complex_handler_logic(self):
        results = []

        def ok_handler(item: dict) -> str:
            results.append("ok")
            return f"Got {item['id']}"

        def not_found_handler(e: NotFound) -> str:
            results.append("not_found")
            return "missing"

        def invalid_handler(e: InvalidInput) -> str:
            results.append("invalid")
            return "bad input"

        match(fetch_item, "test123")(
            {
                Ok: ok_handler,
                NotFound: not_found_handler,
                InvalidInput: invalid_handler,
            }
        )

        assert results == ["ok"]


# Option match tests


@returns_option
def find_item(item_id: str) -> Option[dict]:
    if item_id == "exists":
        return Some({"id": item_id, "name": "Found Item"})
    return Nothing()


class TestOptionMatch:
    def test_dispatches_to_some_handler(self):
        result = match(find_item, "exists")(
            {
                Some: lambda item: f"Found: {item['name']}",
                Nothing: lambda: "not found",
            }
        )
        assert result == "Found: Found Item"

    def test_dispatches_to_nothing_handler(self):
        result = match(find_item, "missing")(
            {
                Some: lambda item: f"Found: {item['name']}",
                Nothing: lambda: "not found",
            }
        )
        assert result == "not found"

    def test_raises_on_missing_some_handler(self):
        with pytest.raises(ExhaustivenessError) as exc_info:
            match(find_item, "exists")(
                {
                    Nothing: lambda: "not found",
                }
            )
        assert Some in exc_info.value.missing

    def test_raises_on_missing_nothing_handler(self):
        with pytest.raises(ExhaustivenessError) as exc_info:
            match(find_item, "exists")(
                {
                    Some: lambda item: item,
                }
            )
        assert Nothing in exc_info.value.missing

    def test_raises_on_extra_handler(self):
        with pytest.raises(ExhaustivenessError) as exc_info:
            match(find_item, "exists")(
                {
                    Some: lambda item: item,
                    Nothing: lambda: None,
                    str: lambda s: s,  # Extra handler
                }
            )
        assert str in exc_info.value.extra

    def test_validates_before_execution(self):
        call_count = 0

        @returns_option
        def side_effect() -> Option[str]:
            nonlocal call_count
            call_count += 1
            return Some("done")

        with pytest.raises(ExhaustivenessError):
            match(side_effect)(
                {
                    Some: lambda x: x,
                    # Missing Nothing handler
                }
            )

        assert call_count == 0

    def test_returns_option_validates_return_type(self):
        @returns_option
        def bad_function() -> Option[int]:
            return 42  # type: ignore  # Wrong! Should return Some or Nothing

        with pytest.raises(TypeError, match=r"returned .* instead of Some or Nothing"):
            bad_function()
