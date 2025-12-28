<p align="center">
  <img src="https://raw.githubusercontent.com/4tyone/pyrethrin/main/PyrethrumLogo.png" alt="Pyrethrin Logo" width="120">
</p>

<h1 align="center">Pyrethrin</h1>

<p align="center">
  <strong>Rust and OCaml-style exhaustive error and None handling for Python</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyrethrin/">
    <img src="https://img.shields.io/pypi/v/pyrethrin?color=blue&label=PyPI" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/pyrethrin/">
    <img src="https://img.shields.io/pypi/pyversions/pyrethrin" alt="Python Version">
  </a>
  <a href="https://github.com/4tyone/pyrethrin/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a>
</p>

---

Pyrethrin brings compile-time safety guarantees to Python for two of the most common sources of runtime errors:

Python's flexibility is its greatest strength and its Achilles' heel. There are weird edge-case exceptions everywhere in your code-base, functions receive `None` in the most unexpected places, and you only discover these issues when your app crashes in production at 3 AM.

Languages like Rust and OCaml solved this with `Result` and `Option` types that make error handling explicit and exhaustive. Pyrethrin brings that same peace of mind to Python: **if it compiles, it handles all the errors**. Stop playing whack-a-mole with try/except blocks and `if x is not None` checks scattered throughout your codebase.

**Exceptions** - Declare what exceptions a function can raise with `@raises`, and the static analyzer ensures every caller handles all of them. No more `except Exception` or forgotten error paths.

**None values** - Mark functions that may return nothing with `@returns_option`, forcing callers to explicitly handle the `Some` and `Nothing` cases. No more `AttributeError: 'NoneType' has no attribute` crashes.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Shields](#shields)
- [API Reference](#api-reference)
- [Option Type](#option-type)
- [Combining Decorators](#combining-decorators)
- [Async Support](#async-support)
- [Pattern Matching](#pattern-matching-python-310)
- [Error Codes](#error-codes)
- [Testing](#testing)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **`@raises` decorator** - Declare exceptions a function can throw
- **`@returns_option` decorator** - Mark functions returning optional values
- **`match()` function** - Exhaustive error handling for Result and Option types
- **`Result` type** - `Ok` and `Err` for explicit success/failure
- **`Option` type** - `Some` and `Nothing` for optional values
- **Static analysis** - Catches missing handlers before runtime
- **Full async/await support** - Works with async functions

**Core Principle:** You must use `match()` or native `match-case` to handle Result and Option types. There are no escape hatches like `unwrap()` - this is by design.

---

## Installation

```bash
pip install pyrethrin
```

From source:

```bash
git clone https://github.com/4tyone/pyrethrin
cd pyrethrin
pip install -e .
```

**Requirements:**
- Python 3.11+
- No additional dependencies required (Pyrethrum binary is bundled)

---

## Quick Start

### 1. Declare Exceptions

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

### 2. Handle All Cases

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

### 4. Missing Handlers? Static Analysis Catches It

```python
# ERROR: Result not handled with match
def bad_handler(user_id: str):
    result = get_user(user_id)
    print(result)  # ExhaustivenessError at runtime

# ERROR: Missing handler for InvalidUserId
match(get_user, user_id)({
    Ok: lambda user: user,
    UserNotFound: lambda e: None,
    # Missing: InvalidUserId - caught by static analysis
})
```

---

## Shields

**New in 0.2.0:** Pyrethrin now ships with pre-built "shields" for popular libraries. Shields are drop-in replacements that add explicit exception declarations to library functions, so you get the same exhaustive error handling guarantees for third-party code.

### Available Shields

| Shield | Coverage | Description |
|--------|----------|-------------|
| `pyrethrin.shields.pandas` | File I/O, parsing, data manipulation | `read_csv`, `read_excel`, `read_json`, `concat`, `merge`, `pivot`, etc. |
| `pyrethrin.shields.numpy` | 95%+ of numpy API | Array creation, math, linalg, FFT, random, I/O |
| `pyrethrin.shields.fastapi` | Core FastAPI | `FastAPI`, `APIRouter`, `Request`, `Response`, dependencies |

### Usage

Replace your imports with the shielded version:

```python
# Before - exceptions are implicit
import pandas as pd
df = pd.read_csv("data.csv")  # Can raise OSError, ParserError, ValueError...

# After - exceptions are explicit and must be handled
from pyrethrin.shields import pandas as pd
from pyrethrin import match, Ok

result = match(pd.read_csv, "data.csv")({
    Ok: lambda df: process(df),
    OSError: lambda e: log_error("File not found", e),
    pd.ParserError: lambda e: log_error("Invalid CSV", e),
    ValueError: lambda e: log_error("Bad data", e),
})
```

Shields export everything from the original library, so you can use them as drop-in replacements. Only the functions that can fail are wrapped with `@raises`.

### Exception Discovery with Arbor

Shield exception declarations are generated using [Arbor](https://github.com/4tyone/arbor), a static analysis tool that traverses Python call graphs to discover all possible exceptions. This ensures shields declare the actual exceptions that can occur, not just guesses.

---

## API Reference

### `@raises(*exceptions)`

Declares which exceptions a function can raise.

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
- Raises `UndeclaredExceptionError` for undeclared exceptions

### `@returns_option`

Marks a function as returning an Option type.

```python
@returns_option
def find_item(items: list, key: str) -> Option[Any]:
    for item in items:
        if item.key == key:
            return Some(item)
    return Nothing()
```

**Requirements:**
- Function must return `Some(value)` or `Nothing()`
- Raises `TypeError` otherwise

### `match(fn, *args, **kwargs)`

Creates a match builder for exhaustive error handling.

```python
# For @raises functions
result = match(risky_function, "key")({
    Ok: lambda value: f"Got {value}",
    ValueError: lambda e: f"Bad value: {e}",
    KeyError: lambda e: f"Missing key: {e}",
})

# For @returns_option functions
result = match(find_item, items, "key")({
    Some: lambda item: f"Found {item}",
    Nothing: lambda: "Not found",
})
```

**Raises `ExhaustivenessError` if:**
- `Ok` handler is missing (for Result types)
- Any declared exception handler is missing
- `Some` or `Nothing` handler is missing (for Option types)

### `Ok` and `Err`

Result types for success or failure.

```python
from pyrethrin import Ok, Err

result: Ok[int] | Err[ValueError] = Ok(42)
result.value      # 42
result.is_ok()    # True
result.is_err()   # False

error = Err(ValueError("oops"))
error.error       # ValueError("oops")
error.is_ok()     # False
error.is_err()    # True
```

### `Some` and `Nothing`

Option types for optional values.

```python
from pyrethrin import Some, Nothing

option = Some(42)
option.value        # 42
option.is_some()    # True
option.is_nothing() # False

empty = Nothing()
empty.is_some()     # False
empty.is_nothing()  # True

Nothing() == Nothing()  # True (all Nothing instances are equal)
```

**Note:** There is no `unwrap()` method. You must use pattern matching.

---

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

# Must handle both cases
result = match(find_user, "123")({
    Some: lambda user: f"Found: {user['name']}",
    Nothing: lambda: "User not found",
})
```

---

## Combining Decorators

Sometimes you need both: a function that may return nothing (`Option`) AND may fail with an exception (`Result`). You can combine `@raises` and `@returns_option` for this.

### Correct Order: `@raises` on top

```python
from pyrethrin import raises, Ok, Err
from pyrethrin.decorators import returns_option
from pyrethrin.option import Some, Nothing, Option

class DatabaseError(Exception):
    pass

class ValidationError(Exception):
    pass

@raises(DatabaseError, ValidationError)
@returns_option
def find_product(product_id: str) -> Option[dict]:
    """
    Find a product by ID.

    Returns:
    - Ok(Some(product)) - found
    - Ok(Nothing()) - not found
    - Err(DatabaseError) - connection failed
    - Err(ValidationError) - invalid ID
    """
    if not product_id.startswith("prod-"):
        raise ValidationError(f"Invalid ID: {product_id}")
    if product_id == "prod-error":
        raise DatabaseError("Connection failed")

    product = PRODUCTS.get(product_id)
    if product is None:
        return Nothing()
    return Some(product)
```

The result type is `Result[Option[T], E]` - a nested type requiring two levels of handling.

### Handling Nested Result[Option[T], E]

**Nested pattern matching (explicit):**

```python
def get_product_price(product_id: str) -> str:
    result = find_product(product_id)

    match result:
        case Ok(option_value):
            match option_value:
                case Some(product):
                    return f"${product['price']:.2f}"
                case Nothing():
                    return "Product not found"
        case Err(DatabaseError() as e):
            return f"Database error: {e}"
        case Err(ValidationError() as e):
            return f"Invalid input: {e}"
```

**Flat pattern matching (concise):**

```python
def get_product_price(product_id: str) -> str:
    match find_product(product_id):
        case Ok(Some(product)):
            return f"${product['price']:.2f}"
        case Ok(Nothing()):
            return "Product not found"
        case Err(DatabaseError() as e):
            return f"Database error: {e}"
        case Err(ValidationError() as e):
            return f"Invalid input: {e}"
```

### Wrong Order: `@returns_option` on top

```python
# DON'T DO THIS - will raise TypeError at runtime
@returns_option
@raises(ValueError)
def wrong_order(x: int) -> int:
    if x < 0:
        raise ValueError("negative")
    return x

wrong_order(5)  # TypeError: returned Ok instead of Some or Nothing
```

The `@raises` decorator returns `Ok`/`Err`, but `@returns_option` expects `Some`/`Nothing`.

### When to Use Combined Decorators

Use `@raises` + `@returns_option` when your function has **two distinct failure modes**:

| Scenario | Use |
|----------|-----|
| Value exists or doesn't | `@returns_option` alone |
| Operation can fail | `@raises` alone |
| Value may not exist AND operation can fail | `@raises` + `@returns_option` |

**Example scenarios:**
- Database lookup that may not find a record AND may have connection errors
- API call that may return no data AND may timeout
- File parsing that may have no matches AND may fail to read

See `examples/combined_decorators_correct.py` and `examples/combined_decorators_missing_handlers.py` for complete examples.

---

## Async Support

Use `@async_raises` and `async_match` for async functions:

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

---

## Pattern Matching (Python 3.10+)

Pyrethrin works with Python's structural pattern matching:

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

The static analyzer verifies exhaustiveness for native `match-case` too.

---

## Error Codes

| Code | Severity | Description |
|------|----------|-------------|
| EXH001 | Error | Missing handlers for declared exceptions |
| EXH002 | Warning | Handlers for undeclared exceptions |
| EXH003 | Error | Missing Ok handler |
| EXH004 | Warning | Unknown function (no @raises signature) |
| EXH005 | Error | Missing Some handler |
| EXH006 | Error | Missing Nothing handler |
| EXH007 | Error | Result not handled with match |
| EXH008 | Error | Option not handled with match |

---

## Exception Types

### `ExhaustivenessError`

Raised when a match is not exhaustive.

```python
from pyrethrin import ExhaustivenessError

try:
    match(get_user, "123")({
        Ok: lambda u: u,
        # Missing exception handlers
    })
except ExhaustivenessError as e:
    print(e.func_name)  # "get_user"
    print(e.missing)    # [UserNotFound, InvalidUserId]
```

### `UndeclaredExceptionError`

Raised when a function raises an exception not in its `@raises` declaration.

```python
from pyrethrin import UndeclaredExceptionError

@raises(ValueError)
def buggy():
    raise KeyError("oops")  # Not declared

try:
    buggy()
except UndeclaredExceptionError as e:
    print(e.fn)        # "buggy"
    print(e.got)       # "KeyError"
    print(e.declared)  # ["ValueError"]
```

---

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

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PYRETHRIN_DISABLE_STATIC_CHECK` | Set to `1` to disable static analysis (useful for production) |

---

## Architecture

Pyrethrin consists of two components:

1. **Python Library** - Runtime decorators, Result/Option types, AST extraction
2. **[Pyrethrum](https://github.com/4tyone/pyrethrum)** - OCaml static analyzer for exhaustiveness checking (bundled as platform-specific binary)

When a decorated function is called:

1. The decorator invokes static analysis on the caller's source file
2. AST is extracted and converted to JSON
3. JSON is passed to the Pyrethrum binary
4. Pyrethrum checks exhaustiveness and returns diagnostics
5. `ExhaustivenessError` is raised if violations are found

Results are cached per call site to avoid redundant analysis.

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

```bash
# Setup development environment
git clone https://github.com/4tyone/pyrethrin
cd pyrethrin
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

---

## License

Apache-2.0