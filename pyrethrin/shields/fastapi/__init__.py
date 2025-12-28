"""Pyrethrin shields for FastAPI.

This module provides shielded versions of FastAPI components with explicit
error handling through the @raises and @async_raises decorators. All functions
that can raise exceptions are wrapped to return Result types, requiring
exhaustive error handling.

Shield version: 0.1.0
FastAPI version: 0.128.0

Usage:
    from pyrethrin.shields.fastapi import (
        create_app,
        create_router,
        request_json,
        request_form,
        websocket_accept,
        websocket_receive_text,
    )
    from pyrethrin import Ok, Err

    # Create app with error handling
    match create_app(title="My API"):
        case Ok(app):
            pass  # use app
        case Err(e):
            print(f"Failed to create app: {e}")

    # In an endpoint, handle request body
    @app.post("/items")
    async def create_item(request: Request):
        match await request_json(request):
            case Ok(data):
                return {"received": data}
            case Err(RuntimeError() as e):
                return {"error": "Stream consumed"}
            case Err(ClientDisconnect() as e):
                return {"error": "Client disconnected"}
"""

from __future__ import annotations

__shield_version__ = "0.1.0"
__fastapi_version__ = "0.128.0"

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Callable, TypeVar

import fastapi as _fastapi
from starlette.datastructures import FormData
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import ClientDisconnect
from starlette.routing import BaseRoute
from starlette.types import ASGIApp, Lifespan

from pyrethrin.async_support import async_raises
from pyrethrin.decorators import raises, returns_option
from pyrethrin.option import Nothing, Some

T = TypeVar("T")

# =============================================================================
# EXCEPTIONS - Re-export all FastAPI/Starlette exceptions
# =============================================================================

from fastapi import HTTPException as HTTPException
from fastapi import WebSocketDisconnect as WebSocketDisconnect
from fastapi import WebSocketException as WebSocketException
from fastapi.exceptions import (
    DependencyScopeError as DependencyScopeError,
)
from fastapi.exceptions import (
    FastAPIError as FastAPIError,
)
from fastapi.exceptions import (
    RequestValidationError as RequestValidationError,
)
from fastapi.exceptions import (
    ResponseValidationError as ResponseValidationError,
)
from fastapi.exceptions import (
    ValidationException as ValidationException,
)
from fastapi.exceptions import (
    WebSocketRequestValidationError as WebSocketRequestValidationError,
)

# Starlette exceptions used by FastAPI
ClientDisconnect = ClientDisconnect
MultiPartException = MultiPartException
StarletteHTTPException = StarletteHTTPException

# =============================================================================
# PARAMETER FUNCTIONS - Re-export unchanged (declarative, low-risk)
# =============================================================================

from fastapi import Body as Body
from fastapi import Cookie as Cookie
from fastapi import Depends as Depends
from fastapi import File as File
from fastapi import Form as Form
from fastapi import Header as Header
from fastapi import Path as Path
from fastapi import Query as Query
from fastapi import Security as Security

# =============================================================================
# DATA STRUCTURES - Re-export types unchanged
# =============================================================================

from fastapi import BackgroundTasks as BackgroundTasks
from fastapi import Request as Request
from fastapi import Response as Response
from fastapi import UploadFile as UploadFile
from fastapi import WebSocket as WebSocket

# Type aliases
FastAPIApp = _fastapi.FastAPI
APIRouterType = _fastapi.APIRouter

# =============================================================================
# STATUS CODES - Re-export unchanged
# =============================================================================

from fastapi import status as status

# =============================================================================
# RESPONSE CLASSES - Re-export unchanged
# =============================================================================

from fastapi.responses import FileResponse as FileResponse
from fastapi.responses import HTMLResponse as HTMLResponse
from fastapi.responses import JSONResponse as JSONResponse
from fastapi.responses import ORJSONResponse as ORJSONResponse
from fastapi.responses import PlainTextResponse as PlainTextResponse
from fastapi.responses import RedirectResponse as RedirectResponse
from fastapi.responses import Response as ResponseBase
from fastapi.responses import StreamingResponse as StreamingResponse
from fastapi.responses import UJSONResponse as UJSONResponse

# =============================================================================
# SECURITY CLASSES - Re-export unchanged
# =============================================================================

from fastapi.security import APIKeyCookie as APIKeyCookie
from fastapi.security import APIKeyHeader as APIKeyHeader
from fastapi.security import APIKeyQuery as APIKeyQuery
from fastapi.security import HTTPAuthorizationCredentials as HTTPAuthorizationCredentials
from fastapi.security import HTTPBasic as HTTPBasic
from fastapi.security import HTTPBasicCredentials as HTTPBasicCredentials
from fastapi.security import HTTPBearer as HTTPBearer
from fastapi.security import HTTPDigest as HTTPDigest
from fastapi.security import OAuth2 as OAuth2
from fastapi.security import OAuth2AuthorizationCodeBearer as OAuth2AuthorizationCodeBearer
from fastapi.security import OAuth2PasswordBearer as OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm as OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordRequestFormStrict as OAuth2PasswordRequestFormStrict
from fastapi.security import OpenIdConnect as OpenIdConnect
from fastapi.security import SecurityScopes as SecurityScopes

# =============================================================================
# MIDDLEWARE - Re-export unchanged
# =============================================================================

from fastapi.middleware import Middleware as Middleware

# =============================================================================
# APP AND ROUTER CREATION - Shielded factory functions
# =============================================================================


@raises(FastAPIError, ValueError, TypeError, RuntimeError)
def create_app(
    *,
    debug: bool = False,
    routes: list[BaseRoute] | None = None,
    title: str = "FastAPI",
    summary: str | None = None,
    description: str = "",
    version: str = "0.1.0",
    openapi_url: str | None = "/openapi.json",
    openapi_tags: list[dict[str, Any]] | None = None,
    servers: list[dict[str, Any]] | None = None,
    dependencies: Sequence[_fastapi.params.Depends] | None = None,
    default_response_class: type[Response] = JSONResponse,
    redirect_slashes: bool = True,
    docs_url: str | None = "/docs",
    redoc_url: str | None = "/redoc",
    swagger_ui_oauth2_redirect_url: str | None = "/docs/oauth2-redirect",
    swagger_ui_init_oauth: dict[str, Any] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: dict[Any, Callable[..., Any]] | None = None,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[FastAPIApp] | None = None,
    terms_of_service: str | None = None,
    contact: dict[str, Any] | None = None,
    license_info: dict[str, Any] | None = None,
    openapi_prefix: str = "",
    root_path: str = "",
    root_path_in_servers: bool = True,
    responses: dict[int | str, dict[str, Any]] | None = None,
    callbacks: list[BaseRoute] | None = None,
    webhooks: _fastapi.routing.APIRouter | None = None,
    deprecated: bool | None = None,
    include_in_schema: bool = True,
    swagger_ui_parameters: dict[str, Any] | None = None,
    generate_unique_id_function: Callable[[_fastapi.routing.APIRoute], str] = _fastapi.utils.generate_unique_id,
    separate_input_output_schemas: bool = True,
    **extra: Any,
) -> FastAPIApp:
    """Create a new FastAPI application instance.

    Returns Result[FastAPI, FastAPIError | ValueError | TypeError | RuntimeError]
    """
    return _fastapi.FastAPI(
        debug=debug,
        routes=routes,
        title=title,
        summary=summary,
        description=description,
        version=version,
        openapi_url=openapi_url,
        openapi_tags=openapi_tags,
        servers=servers,
        dependencies=dependencies,
        default_response_class=default_response_class,
        redirect_slashes=redirect_slashes,
        docs_url=docs_url,
        redoc_url=redoc_url,
        swagger_ui_oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
        swagger_ui_init_oauth=swagger_ui_init_oauth,
        middleware=middleware,
        exception_handlers=exception_handlers,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
        terms_of_service=terms_of_service,
        contact=contact,
        license_info=license_info,
        openapi_prefix=openapi_prefix,
        root_path=root_path,
        root_path_in_servers=root_path_in_servers,
        responses=responses,
        callbacks=callbacks,
        webhooks=webhooks,
        deprecated=deprecated,
        include_in_schema=include_in_schema,
        swagger_ui_parameters=swagger_ui_parameters,
        generate_unique_id_function=generate_unique_id_function,
        separate_input_output_schemas=separate_input_output_schemas,
        **extra,
    )


@raises(FastAPIError, ValueError, TypeError)
def create_router(
    *,
    prefix: str = "",
    tags: list[str | _fastapi.openapi.models.Enum] | None = None,
    dependencies: Sequence[_fastapi.params.Depends] | None = None,
    default_response_class: type[Response] = JSONResponse,
    responses: dict[int | str, dict[str, Any]] | None = None,
    callbacks: list[BaseRoute] | None = None,
    routes: list[BaseRoute] | None = None,
    redirect_slashes: bool = True,
    default: ASGIApp | None = None,
    dependency_overrides_provider: Any | None = None,
    route_class: type[_fastapi.routing.APIRoute] = _fastapi.routing.APIRoute,
    on_startup: Sequence[Callable[[], Any]] | None = None,
    on_shutdown: Sequence[Callable[[], Any]] | None = None,
    lifespan: Lifespan[Any] | None = None,
    deprecated: bool | None = None,
    include_in_schema: bool = True,
    generate_unique_id_function: Callable[[_fastapi.routing.APIRoute], str] = _fastapi.utils.generate_unique_id,
) -> APIRouterType:
    """Create a new APIRouter instance.

    Returns Result[APIRouter, FastAPIError | ValueError | TypeError]
    """
    return _fastapi.APIRouter(
        prefix=prefix,
        tags=tags,
        dependencies=dependencies,
        default_response_class=default_response_class,
        responses=responses,
        callbacks=callbacks,
        routes=routes,
        redirect_slashes=redirect_slashes,
        default=default,
        dependency_overrides_provider=dependency_overrides_provider,
        route_class=route_class,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        lifespan=lifespan,
        deprecated=deprecated,
        include_in_schema=include_in_schema,
        generate_unique_id_function=generate_unique_id_function,
    )


# =============================================================================
# REQUEST METHODS - Shielded async functions
# =============================================================================


@async_raises(RuntimeError, ClientDisconnect)
async def request_body(request: Request) -> bytes:
    """Get the request body as bytes.

    Returns Result[bytes, RuntimeError | ClientDisconnect]

    Raises:
        RuntimeError: If the stream has already been consumed
        ClientDisconnect: If the client disconnected
    """
    return await request.body()


@async_raises(RuntimeError, ClientDisconnect)
async def request_json(request: Request) -> Any:
    """Parse the request body as JSON.

    Returns Result[Any, RuntimeError | ClientDisconnect]

    Raises:
        RuntimeError: If the stream has already been consumed
        ClientDisconnect: If the client disconnected
    """
    return await request.json()


@async_raises(RuntimeError, ClientDisconnect, HTTPException, MultiPartException, KeyError, TypeError)
async def request_form(
    request: Request,
    *,
    max_files: int | float = 1000,
    max_fields: int | float = 1000,
) -> FormData:
    """Parse the request body as form data.

    Returns Result[FormData, RuntimeError | ClientDisconnect | HTTPException | MultiPartException | KeyError | TypeError]

    Raises:
        RuntimeError: If the stream has already been consumed
        ClientDisconnect: If the client disconnected
        HTTPException: If the content type is not supported
        MultiPartException: If multipart parsing fails
    """
    return await request.form(max_files=max_files, max_fields=max_fields)


@async_raises(RuntimeError, ClientDisconnect)
async def request_stream(request: Request) -> AsyncIterator[bytes]:
    """Stream the request body.

    Returns Result[AsyncIterator[bytes], RuntimeError | ClientDisconnect]

    Raises:
        RuntimeError: If the body has already been read
        ClientDisconnect: If the client disconnected
    """
    return request.stream()


@async_raises(RuntimeError)
async def request_close(request: Request) -> None:
    """Close the request.

    Returns Result[None, RuntimeError]
    """
    return await request.close()


@async_raises(RuntimeError)
async def request_is_disconnected(request: Request) -> bool:
    """Check if the client is disconnected.

    Returns Result[bool, RuntimeError]
    """
    return await request.is_disconnected()


@async_raises(RuntimeError)
async def request_send_push_promise(request: Request, path: str) -> None:
    """Send a push promise (HTTP/2).

    Returns Result[None, RuntimeError]
    """
    return await request.send_push_promise(path)


# =============================================================================
# WEBSOCKET METHODS - Shielded async functions
# =============================================================================


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_accept(
    websocket: WebSocket,
    subprotocol: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> None:
    """Accept the WebSocket connection.

    Returns Result[None, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If called in invalid state
        WebSocketDisconnect: If client disconnected during accept
    """
    if headers:
        # Convert to iterable of tuples
        return await websocket.accept(subprotocol=subprotocol, headers=list(headers.items()))
    return await websocket.accept(subprotocol=subprotocol)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_receive(websocket: WebSocket) -> dict[str, Any]:
    """Receive a raw WebSocket message.

    Returns Result[dict, RuntimeError | WebSocketDisconnect]
    """
    return await websocket.receive()


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_receive_text(websocket: WebSocket) -> str:
    """Receive a text message from the WebSocket.

    Returns Result[str, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If not in connected state
        WebSocketDisconnect: If client disconnected
    """
    return await websocket.receive_text()


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_receive_bytes(websocket: WebSocket) -> bytes:
    """Receive a binary message from the WebSocket.

    Returns Result[bytes, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If not in connected state
        WebSocketDisconnect: If client disconnected
    """
    return await websocket.receive_bytes()


@async_raises(RuntimeError, WebSocketDisconnect, ValueError)
async def websocket_receive_json(websocket: WebSocket, mode: str = "text") -> Any:
    """Receive a JSON message from the WebSocket.

    Returns Result[Any, RuntimeError | WebSocketDisconnect | ValueError]

    Raises:
        RuntimeError: If not in connected state
        WebSocketDisconnect: If client disconnected
        ValueError: If JSON parsing fails
    """
    return await websocket.receive_json(mode=mode)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_send(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Send a raw WebSocket message.

    Returns Result[None, RuntimeError | WebSocketDisconnect]
    """
    return await websocket.send(message)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_send_text(websocket: WebSocket, data: str) -> None:
    """Send a text message through the WebSocket.

    Returns Result[None, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If not in connected state
        WebSocketDisconnect: If client disconnected
    """
    return await websocket.send_text(data)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_send_bytes(websocket: WebSocket, data: bytes) -> None:
    """Send a binary message through the WebSocket.

    Returns Result[None, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If not in connected state
        WebSocketDisconnect: If client disconnected
    """
    return await websocket.send_bytes(data)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_send_json(websocket: WebSocket, data: Any, mode: str = "text") -> None:
    """Send a JSON message through the WebSocket.

    Returns Result[None, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If not in connected state or invalid mode
        WebSocketDisconnect: If client disconnected
    """
    return await websocket.send_json(data, mode=mode)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_close(
    websocket: WebSocket, code: int = 1000, reason: str | None = None
) -> None:
    """Close the WebSocket connection.

    Returns Result[None, RuntimeError | WebSocketDisconnect]

    Raises:
        RuntimeError: If called in invalid state
        WebSocketDisconnect: If client already disconnected
    """
    return await websocket.close(code=code, reason=reason)


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_send_denial_response(websocket: WebSocket, response: Response) -> None:
    """Send a denial response before accepting the connection.

    Returns Result[None, RuntimeError | WebSocketDisconnect]
    """
    return await websocket.send_denial_response(response)


# WebSocket iterator wrappers - these return async iterators


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_iter_text(websocket: WebSocket) -> AsyncIterator[str]:
    """Get an async iterator for text messages.

    Returns Result[AsyncIterator[str], RuntimeError | WebSocketDisconnect]
    """
    return websocket.iter_text()


@async_raises(RuntimeError, WebSocketDisconnect)
async def websocket_iter_bytes(websocket: WebSocket) -> AsyncIterator[bytes]:
    """Get an async iterator for binary messages.

    Returns Result[AsyncIterator[bytes], RuntimeError | WebSocketDisconnect]
    """
    return websocket.iter_bytes()


@async_raises(RuntimeError, WebSocketDisconnect, ValueError)
async def websocket_iter_json(websocket: WebSocket) -> AsyncIterator[Any]:
    """Get an async iterator for JSON messages.

    Returns Result[AsyncIterator[Any], RuntimeError | WebSocketDisconnect | ValueError]
    """
    return websocket.iter_json()


# =============================================================================
# UPLOADFILE METHODS - Shielded async functions
# =============================================================================


@async_raises(IOError, ValueError)
async def uploadfile_read(file: UploadFile, size: int = -1) -> bytes:
    """Read bytes from the uploaded file.

    Returns Result[bytes, IOError | ValueError]

    Args:
        file: The UploadFile instance
        size: Number of bytes to read (-1 for all)
    """
    return await file.read(size)


@async_raises(IOError)
async def uploadfile_write(file: UploadFile, data: bytes) -> int:
    """Write bytes to the uploaded file.

    Returns Result[int, IOError]

    Args:
        file: The UploadFile instance
        data: Bytes to write

    Returns the number of bytes written.
    """
    return await file.write(data)


@async_raises(IOError)
async def uploadfile_seek(file: UploadFile, offset: int) -> int:
    """Seek to a position in the uploaded file.

    Returns Result[int, IOError]

    Args:
        file: The UploadFile instance
        offset: Position to seek to
    """
    return await file.seek(offset)


@async_raises(IOError)
async def uploadfile_close(file: UploadFile) -> None:
    """Close the uploaded file.

    Returns Result[None, IOError]
    """
    return await file.close()


# =============================================================================
# ENCODERS - Shielded functions
# =============================================================================


@raises(ValueError, TypeError)
@returns_option
def jsonable_encoder(
    obj: Any,
    include: set[int | str] | Mapping[int | str, Any] | None = None,
    exclude: set[int | str] | Mapping[int | str, Any] | None = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    custom_encoder: dict[Any, Callable[[Any], Any]] | None = None,
    sqlalchemy_safe: bool = True,
) -> Any:
    """Convert a value to a JSON-encodable format.

    Returns Result[Option[Any], ValueError | TypeError]

    - Ok(Some(value)) - successfully encoded
    - Ok(Nothing()) - input was PydanticUndefinedType
    - Err(ValueError | TypeError) - encoding failed

    This is a shielded version of fastapi.encoders.jsonable_encoder.
    """
    result = _fastapi.encoders.jsonable_encoder(
        obj,
        include=include,
        exclude=exclude,
        by_alias=by_alias,
        exclude_unset=exclude_unset,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        custom_encoder=custom_encoder or {},
        sqlalchemy_safe=sqlalchemy_safe,
    )
    if result is None:
        return Nothing()
    return Some(result)


# =============================================================================
# BACKGROUND TASKS - Shielded methods
# =============================================================================


@raises(TypeError)
def background_add_task(
    background_tasks: BackgroundTasks,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Add a background task.

    Returns Result[None, TypeError]

    Args:
        background_tasks: The BackgroundTasks instance
        func: The function to run in the background
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    """
    return background_tasks.add_task(func, *args, **kwargs)


# =============================================================================
# ROUTER METHODS - Shielded functions for common router operations
# =============================================================================


@raises(FastAPIError, ValueError, TypeError)
def router_include_router(
    router: APIRouterType,
    other: APIRouterType,
    *,
    prefix: str = "",
    tags: list[str | _fastapi.openapi.models.Enum] | None = None,
    dependencies: Sequence[_fastapi.params.Depends] | None = None,
    responses: dict[int | str, dict[str, Any]] | None = None,
    deprecated: bool | None = None,
    include_in_schema: bool = True,
    default_response_class: type[Response] = JSONResponse,
    callbacks: list[BaseRoute] | None = None,
    generate_unique_id_function: Callable[[_fastapi.routing.APIRoute], str] = _fastapi.utils.generate_unique_id,
) -> None:
    """Include another router in this router.

    Returns Result[None, FastAPIError | ValueError | TypeError]
    """
    return router.include_router(
        other,
        prefix=prefix,
        tags=tags,
        dependencies=dependencies,
        responses=responses,
        deprecated=deprecated,
        include_in_schema=include_in_schema,
        default_response_class=default_response_class,
        callbacks=callbacks,
        generate_unique_id_function=generate_unique_id_function,
    )


@raises(FastAPIError, ValueError, TypeError)
def app_include_router(
    app: FastAPIApp,
    router: APIRouterType,
    *,
    prefix: str = "",
    tags: list[str | _fastapi.openapi.models.Enum] | None = None,
    dependencies: Sequence[_fastapi.params.Depends] | None = None,
    responses: dict[int | str, dict[str, Any]] | None = None,
    deprecated: bool | None = None,
    include_in_schema: bool = True,
    default_response_class: type[Response] = JSONResponse,
    callbacks: list[BaseRoute] | None = None,
    generate_unique_id_function: Callable[[_fastapi.routing.APIRoute], str] = _fastapi.utils.generate_unique_id,
) -> None:
    """Include a router in the FastAPI app.

    Returns Result[None, FastAPIError | ValueError | TypeError]
    """
    return app.include_router(
        router,
        prefix=prefix,
        tags=tags,
        dependencies=dependencies,
        responses=responses,
        deprecated=deprecated,
        include_in_schema=include_in_schema,
        default_response_class=default_response_class,
        callbacks=callbacks,
        generate_unique_id_function=generate_unique_id_function,
    )


@raises(ValueError, TypeError)
def app_add_middleware(
    app: FastAPIApp,
    middleware_class: type[Any],
    **options: Any,
) -> None:
    """Add middleware to the FastAPI app.

    Returns Result[None, ValueError | TypeError]
    """
    return app.add_middleware(middleware_class, **options)


@raises(ValueError)
def app_mount(
    app: FastAPIApp,
    path: str,
    app_to_mount: ASGIApp,
    name: str | None = None,
) -> None:
    """Mount a sub-application.

    Returns Result[None, ValueError]
    """
    return app.mount(path, app=app_to_mount, name=name)


# =============================================================================
# OPENAPI - Shielded functions
# =============================================================================


@raises(FastAPIError, ValueError, RuntimeError)
def app_openapi(app: FastAPIApp) -> dict[str, Any]:
    """Get the OpenAPI schema.

    Returns Result[dict, FastAPIError | ValueError | RuntimeError]
    """
    return app.openapi()


# =============================================================================
# CONCURRENCY UTILITIES - Shielded async functions
# =============================================================================


@async_raises(Exception)
async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a sync function in a threadpool.

    Returns Result[T, Exception]

    This is useful for running blocking I/O operations without blocking the event loop.
    """
    from fastapi.concurrency import run_in_threadpool as _run_in_threadpool

    return await _run_in_threadpool(func, *args, **kwargs)


@async_raises(Exception)
async def run_until_first_complete(*args: tuple[Callable[[], Any], dict[str, Any]]) -> None:
    """Run multiple async functions until one completes.

    Returns Result[None, Exception]
    """
    from fastapi.concurrency import run_until_first_complete as _run_until_first_complete

    return await _run_until_first_complete(*args)


# iterate_in_threadpool and contextmanager_in_threadpool are async generators/context managers
# Re-export them directly as they require special handling
from fastapi.concurrency import contextmanager_in_threadpool as contextmanager_in_threadpool
from fastapi.concurrency import iterate_in_threadpool as iterate_in_threadpool


# =============================================================================
# ROUTING CLASSES - Re-export unchanged
# =============================================================================

from fastapi.routing import APIRoute as APIRoute
from fastapi.routing import APIWebSocketRoute as APIWebSocketRoute
from starlette.routing import Mount as Mount


# =============================================================================
# OPENAPI UTILITIES - Shielded functions
# =============================================================================


@raises(FastAPIError, ValueError, TypeError, RuntimeError)
def get_openapi(
    *,
    title: str,
    version: str,
    openapi_version: str = "3.1.0",
    summary: str | None = None,
    description: str | None = None,
    routes: Sequence[BaseRoute],
    webhooks: Sequence[BaseRoute] | None = None,
    tags: list[dict[str, Any]] | None = None,
    servers: list[dict[str, Any]] | None = None,
    terms_of_service: str | None = None,
    contact: dict[str, Any] | None = None,
    license_info: dict[str, Any] | None = None,
    separate_input_output_schemas: bool = True,
) -> dict[str, Any]:
    """Generate OpenAPI schema.

    Returns Result[dict, FastAPIError | ValueError | TypeError | RuntimeError]
    """
    from fastapi.openapi.utils import get_openapi as _get_openapi

    return _get_openapi(
        title=title,
        version=version,
        openapi_version=openapi_version,
        summary=summary,
        description=description,
        routes=routes,
        webhooks=webhooks,
        tags=tags,
        servers=servers,
        terms_of_service=terms_of_service,
        contact=contact,
        license_info=license_info,
        separate_input_output_schemas=separate_input_output_schemas,
    )


# =============================================================================
# EXCEPTION HANDLERS - Re-export unchanged (these are meant to be used as handlers)
# =============================================================================

from fastapi.exception_handlers import http_exception_handler as http_exception_handler
from fastapi.exception_handlers import (
    request_validation_exception_handler as request_validation_exception_handler,
)
from fastapi.exception_handlers import (
    websocket_request_validation_exception_handler as websocket_request_validation_exception_handler,
)


# =============================================================================
# STATIC FILES - Re-export unchanged
# =============================================================================

from fastapi.staticfiles import StaticFiles as StaticFiles


# =============================================================================
# TEMPLATING - Re-export unchanged
# =============================================================================

from fastapi.templating import Jinja2Templates as Jinja2Templates


# =============================================================================
# TESTCLIENT - Conditional import (requires httpx)
# =============================================================================

# TestClient requires httpx. We provide a lazy import to avoid breaking
# the module if httpx is not installed.


def __getattr__(name: str) -> Any:
    if name == "TestClient":
        try:
            from fastapi.testclient import TestClient

            return TestClient
        except (ImportError, RuntimeError):
            raise ImportError(
                "TestClient requires httpx. Install it with: pip install httpx"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# ORIGINAL CLASSES - Escape hatch for direct access
# =============================================================================

# These are the unmodified original classes for when you need direct access
FastAPI = _fastapi.FastAPI
APIRouter = _fastapi.APIRouter


# =============================================================================
# __all__ - Public API
# =============================================================================

__all__ = [
    # Version info
    "__shield_version__",
    "__fastapi_version__",
    # Shielded factory functions
    "create_app",
    "create_router",
    # Shielded Request methods
    "request_body",
    "request_json",
    "request_form",
    "request_stream",
    "request_close",
    "request_is_disconnected",
    "request_send_push_promise",
    # Shielded WebSocket methods
    "websocket_accept",
    "websocket_receive",
    "websocket_receive_text",
    "websocket_receive_bytes",
    "websocket_receive_json",
    "websocket_send",
    "websocket_send_text",
    "websocket_send_bytes",
    "websocket_send_json",
    "websocket_close",
    "websocket_send_denial_response",
    "websocket_iter_text",
    "websocket_iter_bytes",
    "websocket_iter_json",
    # Shielded UploadFile methods
    "uploadfile_read",
    "uploadfile_write",
    "uploadfile_seek",
    "uploadfile_close",
    # Shielded encoders
    "jsonable_encoder",
    # Shielded BackgroundTasks
    "background_add_task",
    # Shielded router/app methods
    "router_include_router",
    "app_include_router",
    "app_add_middleware",
    "app_mount",
    "app_openapi",
    # Shielded concurrency utilities
    "run_in_threadpool",
    "run_until_first_complete",
    "iterate_in_threadpool",
    "contextmanager_in_threadpool",
    # Shielded OpenAPI utilities
    "get_openapi",
    # Type aliases
    "FastAPIApp",
    "APIRouterType",
    # Original classes (escape hatch)
    "FastAPI",
    "APIRouter",
    # Routing classes
    "APIRoute",
    "APIWebSocketRoute",
    "Mount",
    # Exceptions
    "HTTPException",
    "WebSocketException",
    "WebSocketDisconnect",
    "FastAPIError",
    "RequestValidationError",
    "ResponseValidationError",
    "ValidationException",
    "WebSocketRequestValidationError",
    "DependencyScopeError",
    "ClientDisconnect",
    "MultiPartException",
    "StarletteHTTPException",
    # Parameter functions
    "Body",
    "Cookie",
    "Depends",
    "File",
    "Form",
    "Header",
    "Path",
    "Query",
    "Security",
    # Data structures
    "BackgroundTasks",
    "Request",
    "Response",
    "UploadFile",
    "WebSocket",
    # Response classes
    "ResponseBase",
    "JSONResponse",
    "HTMLResponse",
    "PlainTextResponse",
    "RedirectResponse",
    "StreamingResponse",
    "FileResponse",
    "ORJSONResponse",
    "UJSONResponse",
    # Security classes
    "APIKeyCookie",
    "APIKeyHeader",
    "APIKeyQuery",
    "HTTPAuthorizationCredentials",
    "HTTPBasic",
    "HTTPBasicCredentials",
    "HTTPBearer",
    "HTTPDigest",
    "OAuth2",
    "OAuth2AuthorizationCodeBearer",
    "OAuth2PasswordBearer",
    "OAuth2PasswordRequestForm",
    "OAuth2PasswordRequestFormStrict",
    "OpenIdConnect",
    "SecurityScopes",
    # Middleware
    "Middleware",
    # Status codes
    "status",
    # Exception handlers
    "http_exception_handler",
    "request_validation_exception_handler",
    "websocket_request_validation_exception_handler",
    # Static files
    "StaticFiles",
    # Templating
    "Jinja2Templates",
    # Testing (requires httpx)
    "TestClient",
    # Option types (for jsonable_encoder)
    "Some",
    "Nothing",
]
