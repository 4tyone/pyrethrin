import pytest

from pyrethrin import ExhaustivenessError, Ok, match, raises
from pyrethrin.testing import assert_raises_signature, expect_exhaustive_error


class NotFound(Exception):
    pass


class InvalidInput(Exception):
    pass


@raises(NotFound, InvalidInput)
def fetch(item_id: str) -> dict:
    if item_id == "missing":
        raise NotFound(item_id)
    return {"id": item_id}


class TestExpectExhaustiveError:
    def test_passes_when_error_raised_with_correct_missing(self):
        with expect_exhaustive_error(missing=[InvalidInput]):
            match(fetch, "abc")(
                {
                    Ok: lambda x: x,
                    NotFound: lambda e: None,
                }
            )

    def test_fails_when_no_error_raised(self):
        with pytest.raises(AssertionError, match="Expected ExhaustivenessError was not raised"):
            with expect_exhaustive_error(missing=[InvalidInput]):
                match(fetch, "abc")(
                    {
                        Ok: lambda x: x,
                        NotFound: lambda e: None,
                        InvalidInput: lambda e: None,
                    }
                )

    def test_fails_when_wrong_missing_types(self):
        with pytest.raises(AssertionError, match="Expected missing="):
            with expect_exhaustive_error(missing=[NotFound]):
                match(fetch, "abc")(
                    {
                        Ok: lambda x: x,
                        NotFound: lambda e: None,
                    }
                )


class TestAssertRaisesSignature:
    def test_passes_with_correct_signature(self):
        assert_raises_signature(fetch, {NotFound, InvalidInput})

    def test_fails_with_wrong_signature(self):
        with pytest.raises(AssertionError, match="Expected signature"):
            assert_raises_signature(fetch, {NotFound})

    def test_fails_for_undecorated_function(self):
        def not_decorated() -> None:
            pass

        with pytest.raises(AssertionError, match="not decorated with @raises"):
            assert_raises_signature(not_decorated, {ValueError})
