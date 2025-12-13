# Pyrethrin

OCaml and Rust-style exhaustive exception handling for Python.

## Overview

Pyrethrin brings Rust-style exhaustive error handling to Python. Instead of unchecked exceptions that can crash your program at runtime, Pyrethrin ensures every declared exception is handled through static analysis enforced at runtime.

**Key Features:**
- `@raises` decorator to declare exceptions a function can throw
- `@returns_option` decorator for functions returning optional values
- `match()` function for exhaustive error handling (works with both Result and Option)
- `Result` type (`Ok`/`Err`) for explicit error handling
- `Option` type (`Some`/`Nothing`) for optional values
- Runtime enforcement via Pyrethrum static analyzer
- Full async/await support

**Core Principle:** You MUST use `match()` or native `match-case` to handle Result and Option types. There are no escape hatches like `unwrap()` - this is by design.

## Installation

```bash
pip install pyrethrin
```

Or install from source:

```bash
git clone https://github.com/4tyone/pyrethrin
cd pyrethrin
pip install -e .
```

## Quick Start

### 1. Declare Exceptions with `@raises`

```python
from pyrethrin import raises, match, Ok, Err

class UserNotFound(Exception):
    pass

class InvalidUserId(Exception):
    pass

@raises(UserNotFound, InvalidUserId)
def get_user(user_id: str) -> User:
    if not user_id.isalnum():
        raise InvalidUserId(f"Invalid ID: {user_id}")
    user = db.find(user_id)
    if user is None:
        raise UserNotFound(user_id)
    return user
```

### 2. Handle All Cases with `match()`

```python
def handle_request(user_id: str) -> Response:
    return match(get_user, user_id)({
        Ok: lambda user: Response(200, user.to_dict()),
        UserNotFound: lambda e: Response(404, {"error": str(e)}),
        InvalidUserId: lambda e: Response(400, {"error": str(e)}),
    })
```

### 3. Or Use Native Pattern Matching

```python
def handle_request(user_id: str) -> Response:
    result = get_user(user_id)
    match result:
        case Ok(user):
            return Response(200, user.to_dict())
        case Err(UserNotFound() as e):
            return Response(404, {"error": str(e)})
        case Err(InvalidUserId() as e):
            return Response(400, {"error": str(e)})
```

### 4. What Happens If You Don't Handle?

```python
# ERROR! This will raise ExhaustivenessError at runtime
def bad_handler(user_id: str):
    result = get_user(user_id)  # Result not handled with match!
    print(result)

# ERROR! Missing handler for InvalidUserId
match(get_user, user_id)({
    Ok: lambda user: user,
    UserNotFound: lambda e: None,
    # Missing: InvalidUserId
})
```

## Option Type

For functions that may or may not return a value:

```python
from pyrethrin import returns_option, match, Some, Nothing, Option

@returns_option
def find_user(user_id: str) -> Option[dict]:
    user = db.get(user_id)
    if user is None:
        return Nothing()
    return Some(user)

# MUST handle both cases
result = match(find_user, "123")({
    Some: lambda user: f"Found: {user['name']}",
    Nothing: lambda: "User not found",
})

# Or with native match-case
result = find_user("123")
match result:
    case Some(user):
        print(f"Found: {user['name']}")
    case Nothing():
        print("User not found")
```

## API Reference

### `@raises(*exceptions)`

Decorator that declares which exceptions a function can raise.

```python
@raises(ValueError, KeyError)
def risky_function(x: str) -> int:
    if not x:
        raise ValueError("empty string")
    return data[x]  # may raise KeyError
```

**Behavior:**
- Returns `Ok(value)` on success
- Returns `Err(exception)` for declared exceptions
- Raises `UndeclaredExceptionError` for undeclared exceptions (indicates a bug)

### `@returns_option`

Decorator that marks a function as returning an Option type.

```python
@returns_option
def find_item(items: list, key: str) -> Option[Any]:
    for item in items:
        if item.key == key:
            return Some(item)
    return Nothing()
```

**Requirements:**
- Function MUST return `Some(value)` or `Nothing()`
- Raises `TypeError` if function returns something else

### `match(fn, *args, **kwargs)`

Creates a match builder for exhaustive error handling.

```python
# For @raises decorated functions
result = match(risky_function, "key")({
    Ok: lambda value: f"Got {value}",
    ValueError: lambda e: f"Bad value: {e}",
    KeyError: lambda e: f"Missing key: {e}",
})

# For @returns_option decorated functions
result = match(find_item, items, "key")({
    Some: lambda item: f"Found {item}",
    Nothing: lambda: "Not found",
})
```

**Validation (raises `ExhaustivenessError` if):**
- `Ok` handler is missing (for Result types)
- Any declared exception handler is missing
- `Some` or `Nothing` handler is missing (for Option types)
- Handlers for undeclared exceptions are provided

### `Ok` and `Err` Types

Result types for representing success or failure.

```python
from pyrethrin import Ok, Err

result: Ok[int] | Err[ValueError] = Ok(42)
result.value      # 42
result.is_ok()    # True
result.is_err()   # False

error_result = Err(ValueError("oops"))
error_result.error    # ValueError("oops")
error_result.is_ok()  # False
error_result.is_err() # True
```

**Methods:**
- `is_ok()` - Returns `True` if Ok
- `is_err()` - Returns `True` if Err

**Note:** There is no `unwrap()` method. You MUST use pattern matching.

### `Some` and `Nothing` Types

Option types for representing optional values.

```python
from pyrethrin import Some, Nothing

option = Some(42)
option.value        # 42
option.is_some()    # True
option.is_nothing() # False

empty: Nothing[int] = Nothing()
empty.is_some()     # False
empty.is_nothing()  # True

# All Nothing instances are equal
Nothing() == Nothing()  # True
```

**Methods:**
- `is_some()` - Returns `True` if Some
- `is_nothing()` - Returns `True` if Nothing

**Note:** There is no `unwrap()` method. You MUST use pattern matching.

### Async Support

For async functions, use `@async_raises` and `async_match`:

```python
from pyrethrin import async_raises, async_match, Ok

@async_raises(ConnectionError, TimeoutError)
async def fetch_data(url: str) -> bytes:
    async with session.get(url) as response:
        return await response.read()

async def handle_fetch(url: str) -> str:
    return await async_match(fetch_data, url)({
        Ok: lambda data: data.decode(),
        ConnectionError: lambda e: "Connection failed",
        TimeoutError: lambda e: "Request timed out",
    })
```

## Error Codes

| Code   | Severity | Description |
|--------|----------|-------------|
| EXH001 | Error    | Missing handlers for declared exceptions |
| EXH002 | Warning  | Handlers for undeclared exceptions |
| EXH003 | Error    | Missing Ok handler |
| EXH004 | Warning  | Unknown function (no @raises signature) |
| EXH005 | Error    | Missing Some handler (Option) |
| EXH006 | Error    | Missing Nothing handler (Option) |
| EXH007 | Error    | Result not handled with match |
| EXH008 | Error    | Option not handled with match |

## Exception Types

### `ExhaustivenessError`

Raised when a match statement is not exhaustive.

```python
from pyrethrin import ExhaustivenessError

try:
    match(get_user, "123")({
        Ok: lambda u: u,
        # Missing exception handlers!
    })
except ExhaustivenessError as e:
    print(e.func_name)   # "get_user"
    print(e.missing)     # [UserNotFound, InvalidUserId]
```

### `UndeclaredExceptionError`

Raised when a function raises an exception not in its `@raises` declaration.

```python
from pyrethrin import UndeclaredExceptionError

@raises(ValueError)
def buggy():
    raise KeyError("oops")  # Not declared!

try:
    buggy()
except UndeclaredExceptionError as e:
    print(e.fn)        # "buggy"
    print(e.got)       # "KeyError"
    print(e.declared)  # ["ValueError"]
```

## Pattern Matching (Python 3.10+)

Pyrethrin works seamlessly with Python's structural pattern matching:

```python
result = get_user("123")
match result:
    case Ok(user):
        return {"status": "ok", "user": user.to_dict()}
    case Err(UserNotFound() as e):
        return {"status": "error", "code": 404, "message": str(e)}
    case Err(InvalidUserId() as e):
        return {"status": "error", "code": 400, "message": str(e)}
```

The static analyzer verifies exhaustiveness for native match-case too.

## Testing

```python
from pyrethrin import raises, match, Ok, Err
import pytest

@raises(ValueError)
def parse_int(s: str) -> int:
    return int(s)

def test_parse_int_success():
    result = parse_int("42")
    assert isinstance(result, Ok)
    assert result.value == 42

def test_parse_int_failure():
    result = parse_int("not a number")
    assert isinstance(result, Err)
    assert isinstance(result.error, ValueError)

def test_exhaustive_handling():
    result = match(parse_int, "42")({
        Ok: lambda n: n * 2,
        ValueError: lambda e: 0,
    })
    assert result == 84
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PYRETHRIN_DISABLE_STATIC_CHECK` | Set to `1` to disable static analysis (for performance) |

## Architecture

Pyrethrin consists of two components:

1. **Python Library** (`pyrethrin`): Runtime decorators, Result/Option types, and AST extraction
2. **Pyrethrum Analyzer**: OCaml-based static analyzer (bundled as platform-specific binary)

When a decorated function is called:
1. The decorator invokes static analysis on the caller's source file
2. AST is extracted and converted to JSON
3. JSON is passed to the Pyrethrum binary
4. Pyrethrum checks exhaustiveness and returns diagnostics
5. `ExhaustivenessError` is raised if violations are found

## License

MIT
