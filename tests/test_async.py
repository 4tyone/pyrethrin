import pytest

from pyrethrin import Err, ExhaustivenessError, Ok, async_match, async_raises


class NotFound(Exception):
    pass


class InvalidInput(Exception):
    pass


@async_raises(NotFound, InvalidInput)
async def async_fetch(item_id: str) -> dict:
    if not item_id:
        raise InvalidInput("empty id")
    if item_id == "missing":
        raise NotFound(item_id)
    return {"id": item_id}


class TestAsyncRaises:
    async def test_returns_ok_on_success(self):
        result = await async_fetch("abc123")
        assert isinstance(result, Ok)
        assert result.value["id"] == "abc123"

    async def test_returns_err_on_declared_exception(self):
        result = await async_fetch("missing")
        assert isinstance(result, Err)
        assert isinstance(result.error, NotFound)

    async def test_attaches_metadata(self):
        assert hasattr(async_fetch, "__pyrethrin_signature__")
        assert hasattr(async_fetch, "__pyrethrin_raises__")


class TestAsyncMatch:
    async def test_dispatches_to_ok_handler(self):
        result = await async_match(async_fetch, "abc123")(
            {
                Ok: lambda item: f"Found: {item['id']}",
                NotFound: lambda e: "not found",
                InvalidInput: lambda e: "invalid",
            }
        )
        assert result == "Found: abc123"

    async def test_dispatches_to_error_handler(self):
        result = await async_match(async_fetch, "missing")(
            {
                Ok: lambda item: "found",
                NotFound: lambda e: "not found",
                InvalidInput: lambda e: "invalid",
            }
        )
        assert result == "not found"

    async def test_raises_on_missing_handler(self):
        with pytest.raises(ExhaustivenessError):
            await async_match(async_fetch, "abc")(
                {
                    Ok: lambda item: "found",
                    NotFound: lambda e: "not found",
                }
            )

    async def test_requires_async_raises_decorated_function(self):
        async def not_decorated(x: int) -> int:
            return x

        with pytest.raises(TypeError, match=r"requires a @async_raises decorated function"):
            async_match(not_decorated, 42)
