"""Tests for the Option type (Some/Nothing)."""

import pytest

from pyrethrin import Some, Nothing, Option


class TestSome:
    """Tests for the Some type."""

    def test_value_accessible(self):
        assert Some(42).value == 42
        assert Some("hello").value == "hello"
        assert Some([1, 2, 3]).value == [1, 2, 3]

    def test_is_some_returns_true(self):
        assert Some(42).is_some() is True

    def test_is_nothing_returns_false(self):
        assert Some(42).is_nothing() is False

    def test_pattern_matching(self):
        option: Option[int] = Some(42)
        match option:
            case Some(value):
                assert value == 42
            case Nothing():
                pytest.fail("Should not match Nothing")

    def test_repr(self):
        assert repr(Some(42)) == "Some(42)"
        assert repr(Some("hello")) == "Some('hello')"


class TestNothing:
    """Tests for the Nothing type."""

    def test_is_some_returns_false(self):
        assert Nothing().is_some() is False

    def test_is_nothing_returns_true(self):
        assert Nothing().is_nothing() is True

    def test_pattern_matching(self):
        option: Option[int] = Nothing()
        match option:
            case Some(value):
                pytest.fail("Should not match Some")
            case Nothing():
                pass  # Expected

    def test_repr(self):
        assert repr(Nothing()) == "Nothing()"

    def test_nothing_instances_are_equal(self):
        assert Nothing() == Nothing()

    def test_nothing_hash_is_consistent(self):
        assert hash(Nothing()) == hash(Nothing())


class TestOptionPatternMatching:
    """Tests for Option pattern matching - the only way to handle Option values."""

    def test_some_pattern_extracts_value(self):
        option = Some(42)
        match option:
            case Some(value):
                assert value == 42
            case Nothing():
                pytest.fail("Should match Some")

    def test_nothing_pattern_matches(self):
        option: Option[int] = Nothing()
        matched = False
        match option:
            case Some(_):
                pytest.fail("Should match Nothing")
            case Nothing():
                matched = True
        assert matched

    def test_nested_option_matching(self):
        option: Option[Option[int]] = Some(Some(42))
        match option:
            case Some(Some(value)):
                assert value == 42
            case _:
                pytest.fail("Should match nested Some")

    def test_option_in_dict_matching(self):
        data = {"result": Some(100)}
        match data["result"]:
            case Some(value):
                assert value == 100
            case Nothing():
                pytest.fail("Should match Some")
