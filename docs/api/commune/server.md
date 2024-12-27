# server



Source: `commune/server.py`

## Classes

### Annotated

Add context-specific metadata to a type.

Example: Annotated[int, runtime_check.Unsigned] indicates to the
hypothetical runtime_check module that this type is an unsigned int.
Every other consumer of this type can ignore this metadata and treat
this type as int.

The first argument to Annotated must be a valid type.

Details:

- It's an error to call `Annotated` with less than two arguments.
- Access the metadata via the ``__metadata__`` attribute::

    assert Annotated[int, '$'].__metadata__ == ('$',)

- Nested Annotated types are flattened::

    assert Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

- Instantiating an annotated type is equivalent to instantiating the
underlying type::

    assert Annotated[C, Ann1](5) == C(5)

- Annotated can be used as a generic type alias::

    type Optimized[T] = Annotated[T, runtime.Optimize()]
    # type checker will treat Optimized[int]
    # as equivalent to Annotated[int, runtime.Optimize()]

    type OptimizedList[T] = Annotated[list[T], runtime.Optimize()]
    # type checker will treat OptimizedList[int]
    # as equivalent to Annotated[list[int], runtime.Optimize()]

- Annotated cannot be used with an unpacked TypeVarTuple::

    type Variadic[*Ts] = Annotated[*Ts, Ann1]  # NOT valid

  This would be equivalent to::

    Annotated[T1, T2, T3, ..., Ann1]

  where T1, T2 etc. are TypeVars, which would be invalid, because
  only one type should be passed to Annotated.

### Any

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### BaseHTTPMiddleware



#### Methods

##### `dispatch(self, request: 'Request', call_next: 'RequestResponseEndpoint') -> 'Response'`



Type annotations:
```python
request: Request
call_next: RequestResponseEndpoint
return: Response
```

### BinaryIO

Typed version of the return of open() in binary mode.

#### Methods

##### `close(self) -> None`



Type annotations:
```python
return: None
```

##### `fileno(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `flush(self) -> None`



Type annotations:
```python
return: None
```

##### `isatty(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `read(self, n: int = -1) -> ~AnyStr`



Type annotations:
```python
n: <class 'int'>
return: ~AnyStr
```

##### `readable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `readline(self, limit: int = -1) -> ~AnyStr`



Type annotations:
```python
limit: <class 'int'>
return: ~AnyStr
```

##### `readlines(self, hint: int = -1) -> List[~AnyStr]`



Type annotations:
```python
hint: <class 'int'>
return: typing.List[~AnyStr]
```

##### `seek(self, offset: int, whence: int = 0) -> int`



Type annotations:
```python
offset: <class 'int'>
whence: <class 'int'>
return: <class 'int'>
```

##### `seekable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `tell(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `truncate(self, size: int = None) -> int`



Type annotations:
```python
size: <class 'int'>
return: <class 'int'>
```

##### `writable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `write(self, s: Union[bytes, bytearray]) -> int`



Type annotations:
```python
s: typing.Union[bytes, bytearray]
return: <class 'int'>
```

##### `writelines(self, lines: List[~AnyStr]) -> None`



Type annotations:
```python
lines: typing.List[~AnyStr]
return: None
```

### CORSMiddleware



#### Methods

##### `allow_explicit_origin(headers: 'MutableHeaders', origin: 'str') -> 'None'`



Type annotations:
```python
headers: MutableHeaders
origin: str
return: None
```

##### `is_allowed_origin(self, origin: 'str') -> 'bool'`



Type annotations:
```python
origin: str
return: bool
```

##### `preflight_response(self, request_headers: 'Headers') -> 'Response'`



Type annotations:
```python
request_headers: Headers
return: Response
```

##### `send(self, message: 'Message', send: 'Send', request_headers: 'Headers') -> 'None'`



Type annotations:
```python
message: Message
send: Send
request_headers: Headers
return: None
```

##### `simple_response(self, scope: 'Scope', receive: 'Receive', send: 'Send', request_headers: 'Headers') -> 'None'`



Type annotations:
```python
scope: Scope
receive: Receive
send: Send
request_headers: Headers
return: None
```

### EventSourceResponse

Streaming response that sends data conforming to the SSE (Server-Sent Events) specification.

#### Methods

##### `delete_cookie(self, key: 'str', path: 'str' = '/', domain: 'str | None' = None, secure: 'bool' = False, httponly: 'bool' = False, samesite: "typing.Literal['lax', 'strict', 'none'] | None" = 'lax') -> 'None'`



Type annotations:
```python
key: str
path: str
domain: str | None
secure: bool
httponly: bool
samesite: typing.Literal['lax', 'strict', 'none'] | None
return: None
```

##### `enable_compression(self, force: bool = False) -> None`



Type annotations:
```python
force: <class 'bool'>
return: None
```

##### `init_headers(self, headers: 'typing.Mapping[str, str] | None' = None) -> 'None'`



Type annotations:
```python
headers: typing.Mapping[str, str] | None
return: None
```

##### `render(self, content: 'typing.Any') -> 'bytes | memoryview'`



Type annotations:
```python
content: typing.Any
return: bytes | memoryview
```

##### `set_cookie(self, key: 'str', value: 'str' = '', max_age: 'int | None' = None, expires: 'datetime | str | int | None' = None, path: 'str | None' = '/', domain: 'str | None' = None, secure: 'bool' = False, httponly: 'bool' = False, samesite: "typing.Literal['lax', 'strict', 'none'] | None" = 'lax') -> 'None'`



Type annotations:
```python
key: str
value: str
max_age: int | None
expires: datetime | str | int | None
path: str | None
domain: str | None
secure: bool
httponly: bool
samesite: typing.Literal['lax', 'strict', 'none'] | None
return: None
```

### FastAPI

`FastAPI` app class, the main entrypoint to use FastAPI.

Read more in the
[FastAPI docs for First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/).

## Example

```python
from fastapi import FastAPI

app = FastAPI()
```

#### Methods

##### `add_api_route(self, path: str, endpoint: Callable[..., Any], *, response_model: Any = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce97f50>, status_code: Optional[int] = None, tags: Optional[List[Union[str, enum.Enum]]] = None, dependencies: Optional[Sequence[fastapi.params.Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType] = None, response_model_exclude: Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Union[Type[starlette.responses.Response], fastapi.datastructures.DefaultPlaceholder] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce97f80>, name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Callable[[fastapi.routing.APIRoute], str] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce97fb0>) -> None`



Type annotations:
```python
path: <class 'str'>
endpoint: typing.Callable[..., typing.Any]
response_model: typing.Any
status_code: typing.Optional[int]
tags: typing.Optional[typing.List[typing.Union[str, enum.Enum]]]
dependencies: typing.Optional[typing.Sequence[fastapi.params.Depends]]
summary: typing.Optional[str]
description: typing.Optional[str]
response_description: <class 'str'>
responses: typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]]
deprecated: typing.Optional[bool]
methods: typing.Optional[typing.List[str]]
operation_id: typing.Optional[str]
response_model_include: typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType]
response_model_exclude: typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType]
response_model_by_alias: <class 'bool'>
response_model_exclude_unset: <class 'bool'>
response_model_exclude_defaults: <class 'bool'>
response_model_exclude_none: <class 'bool'>
include_in_schema: <class 'bool'>
response_class: typing.Union[typing.Type[starlette.responses.Response], fastapi.datastructures.DefaultPlaceholder]
name: typing.Optional[str]
openapi_extra: typing.Optional[typing.Dict[str, typing.Any]]
generate_unique_id_function: typing.Callable[[fastapi.routing.APIRoute], str]
return: None
```

##### `add_api_websocket_route(self, path: str, endpoint: Callable[..., Any], name: Optional[str] = None, *, dependencies: Optional[Sequence[fastapi.params.Depends]] = None) -> None`



Type annotations:
```python
path: <class 'str'>
endpoint: typing.Callable[..., typing.Any]
name: typing.Optional[str]
dependencies: typing.Optional[typing.Sequence[fastapi.params.Depends]]
return: None
```

##### `add_event_handler(self, event_type: 'str', func: 'typing.Callable') -> 'None'`



Type annotations:
```python
event_type: str
func: typing.Callable
return: None
```

##### `add_exception_handler(self, exc_class_or_status_code: 'int | type[Exception]', handler: 'ExceptionHandler') -> 'None'`



Type annotations:
```python
exc_class_or_status_code: int | type[Exception]
handler: ExceptionHandler
return: None
```

##### `add_middleware(self, middleware_class: '_MiddlewareFactory[P]', *args: 'P.args', **kwargs: 'P.kwargs') -> 'None'`



Type annotations:
```python
middleware_class: _MiddlewareFactory[P]
args: P.args
kwargs: P.kwargs
return: None
```

##### `add_route(self, path: 'str', route: 'typing.Callable[[Request], typing.Awaitable[Response] | Response]', methods: 'list[str] | None' = None, name: 'str | None' = None, include_in_schema: 'bool' = True) -> 'None'`



Type annotations:
```python
path: str
route: typing.Callable[[Request], typing.Awaitable[Response] | Response]
methods: list[str] | None
name: str | None
include_in_schema: bool
return: None
```

##### `add_websocket_route(self, path: 'str', route: 'typing.Callable[[WebSocket], typing.Awaitable[None]]', name: 'str | None' = None) -> 'None'`



Type annotations:
```python
path: str
route: typing.Callable[[WebSocket], typing.Awaitable[None]]
name: str | None
return: None
```

##### `api_route(self, path: str, *, response_model: Any = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce97fe0>, status_code: Optional[int] = None, tags: Optional[List[Union[str, enum.Enum]]] = None, dependencies: Optional[Sequence[fastapi.params.Depends]] = None, summary: Optional[str] = None, description: Optional[str] = None, response_description: str = 'Successful Response', responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None, deprecated: Optional[bool] = None, methods: Optional[List[str]] = None, operation_id: Optional[str] = None, response_model_include: Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType] = None, response_model_exclude: Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType] = None, response_model_by_alias: bool = True, response_model_exclude_unset: bool = False, response_model_exclude_defaults: bool = False, response_model_exclude_none: bool = False, include_in_schema: bool = True, response_class: Type[starlette.responses.Response] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64050>, name: Optional[str] = None, openapi_extra: Optional[Dict[str, Any]] = None, generate_unique_id_function: Callable[[fastapi.routing.APIRoute], str] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64080>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`



Type annotations:
```python
path: <class 'str'>
response_model: typing.Any
status_code: typing.Optional[int]
tags: typing.Optional[typing.List[typing.Union[str, enum.Enum]]]
dependencies: typing.Optional[typing.Sequence[fastapi.params.Depends]]
summary: typing.Optional[str]
description: typing.Optional[str]
response_description: <class 'str'>
responses: typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]]
deprecated: typing.Optional[bool]
methods: typing.Optional[typing.List[str]]
operation_id: typing.Optional[str]
response_model_include: typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType]
response_model_exclude: typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType]
response_model_by_alias: <class 'bool'>
response_model_exclude_unset: <class 'bool'>
response_model_exclude_defaults: <class 'bool'>
response_model_exclude_none: <class 'bool'>
include_in_schema: <class 'bool'>
response_class: typing.Type[starlette.responses.Response]
name: typing.Optional[str]
openapi_extra: typing.Optional[typing.Dict[str, typing.Any]]
generate_unique_id_function: typing.Callable[[fastapi.routing.APIRoute], str]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `build_middleware_stack(self) -> 'ASGIApp'`



Type annotations:
```python
return: ASGIApp
```

##### `delete(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce646b0>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce646e0>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64710>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP DELETE operation.

## Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.delete("/items/{item_id}")
def delete_item(item_id: str):
    return {"message": "Item deleted"}
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `exception_handler(self, exc_class_or_status_code: Annotated[Union[int, Type[Exception]], Doc('\n                The Exception class this would handle, or a status code.\n                ')]) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add an exception handler to the app.

Read more about it in the
[FastAPI docs for Handling Errors](https://fastapi.tiangolo.com/tutorial/handling-errors/).

## Example

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


app = FastAPI()


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )
```

Type annotations:
```python
exc_class_or_status_code: typing.Annotated[typing.Union[int, typing.Type[Exception]], Doc('\n                The Exception class this would handle, or a status code.\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `get(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64500>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64530>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64560>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP GET operation.

## Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
def read_items():
    return [{"name": "Empanada"}, {"name": "Arepa"}]
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `head(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce647d0>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64800>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64830>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP HEAD operation.

## Example

```python
from fastapi import FastAPI, Response

app = FastAPI()

@app.head("/items/", status_code=204)
def get_items_headers(response: Response):
    response.headers["X-Cat-Dog"] = "Alone in the world"
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `host(self, host: 'str', app: 'ASGIApp', name: 'str | None' = None) -> 'None'`



Type annotations:
```python
host: str
app: ASGIApp
name: str | None
return: None
```

##### `include_router(self, router: Annotated[fastapi.routing.APIRouter, Doc('The `APIRouter` to include.')], *, prefix: Annotated[str, Doc('An optional path prefix for the router.')] = '', tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to all the *path operations* in this\n                router.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to all the\n                *path operations* in this router.\n\n                Read more about it in the\n                [FastAPI docs for Bigger Applications - Multiple Files](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n\n                **Example**\n\n                ```python\n                from fastapi import Depends, FastAPI\n\n                from .dependencies import get_token_header\n                from .internal import admin\n\n                app = FastAPI()\n\n                app.include_router(\n                    admin.router,\n                    dependencies=[Depends(get_token_header)],\n                )\n                ```\n                ')] = None, responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses to be shown in OpenAPI.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Additional Responses in OpenAPI](https://fastapi.tiangolo.com/advanced/additional-responses/).\n\n                And in the\n                [FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark all the *path operations* in this router as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    deprecated=True,\n                )\n                ```\n                ')] = None, include_in_schema: Annotated[bool, Doc('\n                Include (or not) all the *path operations* in this router in the\n                generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    include_in_schema=False,\n                )\n                ```\n                ')] = True, default_response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Default response class to be used for the *path operations* in this\n                router.\n\n                Read more in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#default-response-class).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n                from fastapi.responses import ORJSONResponse\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    default_response_class=ORJSONResponse,\n                )\n                ```\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce640b0>, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce640e0>) -> None`

Include an `APIRouter` in the same app.

Read more about it in the
[FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/).

## Example

```python
from fastapi import FastAPI

from .users import users_router

app = FastAPI()

app.include_router(users_router)
```

Type annotations:
```python
router: typing.Annotated[fastapi.routing.APIRouter, Doc('The `APIRouter` to include.')]
prefix: typing.Annotated[str, Doc('An optional path prefix for the router.')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to all the *path operations* in this\n                router.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to all the\n                *path operations* in this router.\n\n                Read more about it in the\n                [FastAPI docs for Bigger Applications - Multiple Files](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n\n                **Example**\n\n                ```python\n                from fastapi import Depends, FastAPI\n\n                from .dependencies import get_token_header\n                from .internal import admin\n\n                app = FastAPI()\n\n                app.include_router(\n                    admin.router,\n                    dependencies=[Depends(get_token_header)],\n                )\n                ```\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses to be shown in OpenAPI.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Additional Responses in OpenAPI](https://fastapi.tiangolo.com/advanced/additional-responses/).\n\n                And in the\n                [FastAPI docs for Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark all the *path operations* in this router as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    deprecated=True,\n                )\n                ```\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include (or not) all the *path operations* in this router in the\n                generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    include_in_schema=False,\n                )\n                ```\n                ')]
default_response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Default response class to be used for the *path operations* in this\n                router.\n\n                Read more in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#default-response-class).\n\n                **Example**\n\n                ```python\n                from fastapi import FastAPI\n                from fastapi.responses import ORJSONResponse\n\n                from .internal import old_api\n\n                app = FastAPI()\n\n                app.include_router(\n                    old_api.router,\n                    default_response_class=ORJSONResponse,\n                )\n                ```\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: None
```

##### `middleware(self, middleware_type: Annotated[str, Doc('\n                The type of middleware. Currently only supports `http`.\n                ')]) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a middleware to the application.

Read more about it in the
[FastAPI docs for Middleware](https://fastapi.tiangolo.com/tutorial/middleware/).

## Example

```python
import time

from fastapi import FastAPI, Request

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

Type annotations:
```python
middleware_type: typing.Annotated[str, Doc('\n                The type of middleware. Currently only supports `http`.\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `mount(self, path: 'str', app: 'ASGIApp', name: 'str | None' = None) -> 'None'`



Type annotations:
```python
path: str
app: ASGIApp
name: str | None
return: None
```

##### `on_event(self, event_type: Annotated[str, Doc('\n                The type of event. `startup` or `shutdown`.\n                ')]) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add an event handler for the application.

`on_event` is deprecated, use `lifespan` event handlers instead.

Read more about it in the
[FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated).

Type annotations:
```python
event_type: typing.Annotated[str, Doc('\n                The type of event. `startup` or `shutdown`.\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `openapi(self) -> Dict[str, Any]`

Generate the OpenAPI schema of the application. This is called by FastAPI
internally.

The first time it is called it stores the result in the attribute
`app.openapi_schema`, and next times it is called, it just returns that same
result. To avoid the cost of generating the schema every time.

If you need to modify the generated OpenAPI schema, you could modify it.

Read more in the
[FastAPI docs for OpenAPI](https://fastapi.tiangolo.com/how-to/extending-openapi/).

Type annotations:
```python
return: typing.Dict[str, typing.Any]
```

##### `options(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64740>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64770>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce647a0>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP OPTIONS operation.

## Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.options("/items/")
def get_item_options():
    return {"additions": ["Aji", "Guacamole"]}
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `patch(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64860>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64890>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce648c0>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP PATCH operation.

## Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None

app = FastAPI()

@app.patch("/items/")
def update_item(item: Item):
    return {"message": "Item updated in place"}
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `post(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64620>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64650>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64680>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP POST operation.

## Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None

app = FastAPI()

@app.post("/items/")
def create_item(item: Item):
    return {"message": "Item created"}
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `put(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64590>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce645c0>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce645f0>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP PUT operation.

## Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None

app = FastAPI()

@app.put("/items/{item_id}")
def replace_item(item_id: str, item: Item):
    return {"message": "Item replaced", "id": item_id}
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `route(self, path: 'str', methods: 'list[str] | None' = None, name: 'str | None' = None, include_in_schema: 'bool' = True) -> 'typing.Callable'`

We no longer document this decorator style API, and its usage is discouraged.
Instead you should use the following approach:

>>> routes = [Route(path, endpoint=...), ...]
>>> app = Starlette(routes=routes)

Type annotations:
```python
path: str
methods: list[str] | None
name: str | None
include_in_schema: bool
return: typing.Callable
```

##### `setup(self) -> None`



Type annotations:
```python
return: None
```

##### `trace(self, path: Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')], *, response_model: Annotated[Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce648f0>, status_code: Annotated[Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')] = None, tags: Annotated[Optional[List[Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')] = None, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')] = None, summary: Annotated[Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, description: Annotated[Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')] = None, response_description: Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = 'Successful Response', responses: Annotated[Optional[Dict[Union[int, str], Dict[str, Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, deprecated: Annotated[Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')] = None, operation_id: Annotated[Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = None, response_model_include: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_exclude: Annotated[Union[Set[int], Set[str], Dict[int, Any], Dict[str, Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = None, response_model_by_alias: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')] = True, response_model_exclude_unset: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_defaults: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')] = False, response_model_exclude_none: Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')] = False, include_in_schema: Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')] = True, response_class: Annotated[Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64920>, name: Annotated[Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')] = None, callbacks: Annotated[Optional[List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")] = None, openapi_extra: Annotated[Optional[Dict[str, Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')] = None, generate_unique_id_function: Annotated[Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')] = <fastapi.datastructures.DefaultPlaceholder object at 0xfbf9cce64950>) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Add a *path operation* using an HTTP TRACE operation.

## Example

```python
from fastapi import FastAPI

app = FastAPI()

@app.put("/items/{item_id}")
def trace_item(item_id: str):
    return None
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                The URL path to be used for this *path operation*.\n\n                For example, in `http://example.com/items`, the path is `/items`.\n                ')]
response_model: typing.Annotated[typing.Any, Doc("\n                The type to use for the response.\n\n                It could be any valid Pydantic *field* type. So, it doesn't have to\n                be a Pydantic model, it could be other things, like a `list`, `dict`,\n                etc.\n\n                It will be used for:\n\n                * Documentation: the generated OpenAPI (and the UI at `/docs`) will\n                    show it as the response (JSON Schema).\n                * Serialization: you could return an arbitrary object and the\n                    `response_model` would be used to serialize that object into the\n                    corresponding JSON.\n                * Filtering: the JSON sent to the client will only contain the data\n                    (fields) defined in the `response_model`. If you returned an object\n                    that contains an attribute `password` but the `response_model` does\n                    not include that field, the JSON sent to the client would not have\n                    that `password`.\n                * Validation: whatever you return will be serialized with the\n                    `response_model`, converting any data as necessary to generate the\n                    corresponding JSON. But if the data in the object returned is not\n                    valid, that would mean a violation of the contract with the client,\n                    so it's an error from the API developer. So, FastAPI will raise an\n                    error and return a 500 error code (Internal Server Error).\n\n                Read more about it in the\n                [FastAPI docs for Response Model](https://fastapi.tiangolo.com/tutorial/response-model/).\n                ")]
status_code: typing.Annotated[typing.Optional[int], Doc('\n                The default status code to be used for the response.\n\n                You could override the status code by returning a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Response Status Code](https://fastapi.tiangolo.com/tutorial/response-status-code/).\n                ')]
tags: typing.Annotated[typing.Optional[typing.List[typing.Union[str, enum.Enum]]], Doc('\n                A list of tags to be applied to the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/#tags).\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be applied to the\n                *path operation*.\n\n                Read more about it in the\n                [FastAPI docs for Dependencies in path operation decorators](https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-in-path-operation-decorators/).\n                ')]
summary: typing.Annotated[typing.Optional[str], Doc('\n                A summary for the *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
description: typing.Annotated[typing.Optional[str], Doc('\n                A description for the *path operation*.\n\n                If not provided, it will be extracted automatically from the docstring\n                of the *path operation function*.\n\n                It can contain Markdown.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Configuration](https://fastapi.tiangolo.com/tutorial/path-operation-configuration/).\n                ')]
response_description: typing.Annotated[str, Doc('\n                The description for the default response.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
responses: typing.Annotated[typing.Optional[typing.Dict[typing.Union[int, str], typing.Dict[str, typing.Any]]], Doc('\n                Additional responses that could be returned by this *path operation*.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
deprecated: typing.Annotated[typing.Optional[bool], Doc('\n                Mark this *path operation* as deprecated.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n                ')]
operation_id: typing.Annotated[typing.Optional[str], Doc('\n                Custom operation ID to be used by this *path operation*.\n\n                By default, it is generated automatically.\n\n                If you provide a custom operation ID, you need to make sure it is\n                unique for the whole API.\n\n                You can customize the\n                operation ID generation with the parameter\n                `generate_unique_id_function` in the `FastAPI` class.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
response_model_include: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to include only certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude: typing.Annotated[typing.Union[typing.Set[int], typing.Set[str], typing.Dict[int, typing.Any], typing.Dict[str, typing.Any], NoneType], Doc('\n                Configuration passed to Pydantic to exclude certain fields in the\n                response data.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_by_alias: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response model\n                should be serialized by alias when an alias is used.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_include-and-response_model_exclude).\n                ')]
response_model_exclude_unset: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that were not set and\n                have their default values. This is different from\n                `response_model_exclude_defaults` in that if the fields are set,\n                they will be included in the response, even if the value is the same\n                as the default.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_defaults: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data\n                should have all the fields, including the ones that have the same value\n                as the default. This is different from `response_model_exclude_unset`\n                in that if the fields are set but contain the same default values,\n                they will be excluded from the response.\n\n                When `True`, default values are omitted from the response.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#use-the-response_model_exclude_unset-parameter).\n                ')]
response_model_exclude_none: typing.Annotated[bool, Doc('\n                Configuration passed to Pydantic to define if the response data should\n                exclude fields set to `None`.\n\n                This is much simpler (less smart) than `response_model_exclude_unset`\n                and `response_model_exclude_defaults`. You probably want to use one of\n                those two instead of this one, as those allow returning `None` values\n                when it makes sense.\n\n                Read more about it in the\n                [FastAPI docs for Response Model - Return Type](https://fastapi.tiangolo.com/tutorial/response-model/#response_model_exclude_none).\n                ')]
include_in_schema: typing.Annotated[bool, Doc('\n                Include this *path operation* in the generated OpenAPI schema.\n\n                This affects the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for Query Parameters and String Validations](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/#exclude-from-openapi).\n                ')]
response_class: typing.Annotated[typing.Type[starlette.responses.Response], Doc('\n                Response class to be used for this *path operation*.\n\n                This will not be used if you return a response directly.\n\n                Read more about it in the\n                [FastAPI docs for Custom Response - HTML, Stream, File, others](https://fastapi.tiangolo.com/advanced/custom-response/#redirectresponse).\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                Name for this *path operation*. Only used internally.\n                ')]
callbacks: typing.Annotated[typing.Optional[typing.List[starlette.routing.BaseRoute]], Doc("\n                List of *path operations* that will be used as OpenAPI callbacks.\n\n                This is only for OpenAPI documentation, the callbacks won't be used\n                directly.\n\n                It will be added to the generated OpenAPI (e.g. visible at `/docs`).\n\n                Read more about it in the\n                [FastAPI docs for OpenAPI Callbacks](https://fastapi.tiangolo.com/advanced/openapi-callbacks/).\n                ")]
openapi_extra: typing.Annotated[typing.Optional[typing.Dict[str, typing.Any]], Doc('\n                Extra metadata to be included in the OpenAPI schema for this *path\n                operation*.\n\n                Read more about it in the\n                [FastAPI docs for Path Operation Advanced Configuration](https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#custom-openapi-path-operation-schema).\n                ')]
generate_unique_id_function: typing.Annotated[typing.Callable[[fastapi.routing.APIRoute], str], Doc('\n                Customize the function used to generate unique IDs for the *path\n                operations* shown in the generated OpenAPI.\n\n                This is particularly useful when automatically generating clients or\n                SDKs for your API.\n\n                Read more about it in the\n                [FastAPI docs about how to Generate Clients](https://fastapi.tiangolo.com/advanced/generate-clients/#custom-generate-unique-id-function).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `url_path_for(self, name: 'str', /, **path_params: 'typing.Any') -> 'URLPath'`



Type annotations:
```python
name: str
path_params: typing.Any
return: URLPath
```

##### `websocket(self, path: Annotated[str, Doc('\n                WebSocket path.\n                ')], name: Annotated[Optional[str], Doc('\n                A name for the WebSocket. Only used internally.\n                ')] = None, *, dependencies: Annotated[Optional[Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be used for this\n                WebSocket.\n\n                Read more about it in the\n                [FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).\n                ')] = None) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

Decorate a WebSocket function.

Read more about it in the
[FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).

**Example**

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

Type annotations:
```python
path: typing.Annotated[str, Doc('\n                WebSocket path.\n                ')]
name: typing.Annotated[typing.Optional[str], Doc('\n                A name for the WebSocket. Only used internally.\n                ')]
dependencies: typing.Annotated[typing.Optional[typing.Sequence[fastapi.params.Depends]], Doc('\n                A list of dependencies (using `Depends()`) to be used for this\n                WebSocket.\n\n                Read more about it in the\n                [FastAPI docs for WebSockets](https://fastapi.tiangolo.com/advanced/websockets/).\n                ')]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

##### `websocket_route(self, path: str, name: Optional[str] = None) -> Callable[[~DecoratedCallable], ~DecoratedCallable]`

We no longer document this decorator style API, and its usage is discouraged.
Instead you should use the following approach:

>>> routes = [WebSocketRoute(path, endpoint=...), ...]
>>> app = Starlette(routes=routes)

Type annotations:
```python
path: <class 'str'>
name: typing.Optional[str]
return: typing.Callable[[~DecoratedCallable], ~DecoratedCallable]
```

### ForwardRef

Internal wrapper to hold a forward reference.

### Generic

Abstract base class for generic types.

On Python 3.12 and newer, generic classes implicitly inherit from
Generic when they declare a parameter list after the class's name::

    class Mapping[KT, VT]:
        def __getitem__(self, key: KT) -> VT:
            ...
        # Etc.

On older versions of Python, however, generic classes have to
explicitly inherit from Generic.

After a class has been declared to be generic, it can then be used as
follows::

    def lookup_name[KT, VT](mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
        try:
            return mapping[key]
        except KeyError:
            return default

### IO

Generic base class for TextIO and BinaryIO.

This is an abstract, generic version of the return of open().

NOTE: This does not distinguish between the different possible
classes (text vs. binary, read vs. write vs. read/write,
append-only, unbuffered).  The TextIO and BinaryIO subclasses
below capture the distinctions between text vs. binary, which is
pervasive in the interface; however we currently do not offer a
way to track the other distinctions in the type system.

#### Methods

##### `close(self) -> None`



Type annotations:
```python
return: None
```

##### `fileno(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `flush(self) -> None`



Type annotations:
```python
return: None
```

##### `isatty(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `read(self, n: int = -1) -> ~AnyStr`



Type annotations:
```python
n: <class 'int'>
return: ~AnyStr
```

##### `readable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `readline(self, limit: int = -1) -> ~AnyStr`



Type annotations:
```python
limit: <class 'int'>
return: ~AnyStr
```

##### `readlines(self, hint: int = -1) -> List[~AnyStr]`



Type annotations:
```python
hint: <class 'int'>
return: typing.List[~AnyStr]
```

##### `seek(self, offset: int, whence: int = 0) -> int`



Type annotations:
```python
offset: <class 'int'>
whence: <class 'int'>
return: <class 'int'>
```

##### `seekable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `tell(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `truncate(self, size: int = None) -> int`



Type annotations:
```python
size: <class 'int'>
return: <class 'int'>
```

##### `writable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `write(self, s: ~AnyStr) -> int`



Type annotations:
```python
s: ~AnyStr
return: <class 'int'>
```

##### `writelines(self, lines: List[~AnyStr]) -> None`



Type annotations:
```python
lines: typing.List[~AnyStr]
return: None
```

### JSONResponse



#### Methods

##### `delete_cookie(self, key: 'str', path: 'str' = '/', domain: 'str | None' = None, secure: 'bool' = False, httponly: 'bool' = False, samesite: "typing.Literal['lax', 'strict', 'none'] | None" = 'lax') -> 'None'`



Type annotations:
```python
key: str
path: str
domain: str | None
secure: bool
httponly: bool
samesite: typing.Literal['lax', 'strict', 'none'] | None
return: None
```

##### `init_headers(self, headers: 'typing.Mapping[str, str] | None' = None) -> 'None'`



Type annotations:
```python
headers: typing.Mapping[str, str] | None
return: None
```

##### `render(self, content: 'typing.Any') -> 'bytes'`



Type annotations:
```python
content: typing.Any
return: bytes
```

##### `set_cookie(self, key: 'str', value: 'str' = '', max_age: 'int | None' = None, expires: 'datetime | str | int | None' = None, path: 'str | None' = '/', domain: 'str | None' = None, secure: 'bool' = False, httponly: 'bool' = False, samesite: "typing.Literal['lax', 'strict', 'none'] | None" = 'lax') -> 'None'`



Type annotations:
```python
key: str
value: str
max_age: int | None
expires: datetime | str | int | None
path: str | None
domain: str | None
secure: bool
httponly: bool
samesite: typing.Literal['lax', 'strict', 'none'] | None
return: None
```

### Middleware



#### Methods

##### `dispatch(self, request: starlette.requests.Request, call_next)`



Type annotations:
```python
request: <class 'starlette.requests.Request'>
```

### NewType

NewType creates simple unique types with almost zero runtime overhead.

NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument.

Usage::

    UserId = NewType('UserId', int)

    def name_by_id(user_id: UserId) -> str:
        ...

    UserId('user')          # Fails type check

    name_by_id(42)          # Fails type check
    name_by_id(UserId(42))  # OK

    num = UserId(5) + 1     # type: int

### ParamSpec

Parameter specification variable.

The preferred way to construct a parameter specification is via the
dedicated syntax for generic functions, classes, and type aliases,
where the use of '**' creates a parameter specification::

    type IntFunc[**P] = Callable[P, int]

For compatibility with Python 3.11 and earlier, ParamSpec objects
can also be created as follows::

    P = ParamSpec('P')

Parameter specification variables exist primarily for the benefit of
static type checkers.  They are used to forward the parameter types of
one callable to another callable, a pattern commonly found in
higher-order functions and decorators.  They are only valid when used
in ``Concatenate``, or as the first argument to ``Callable``, or as
parameters for user-defined Generics. See class Generic for more
information on generic types.

An example for annotating a decorator::

    def add_logging[**P, T](f: Callable[P, T]) -> Callable[P, T]:
        '''A type-safe decorator to add logging to a function.'''
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            logging.info(f'{f.__name__} was called')
            return f(*args, **kwargs)
        return inner

    @add_logging
    def add_two(x: float, y: float) -> float:
        '''Add two numbers together.'''
        return x + y

Parameter specification variables can be introspected. e.g.::

    >>> P = ParamSpec("P")
    >>> P.__name__
    'P'

Note that only parameter specification variables defined in the global
scope can be pickled.

### ParamSpecArgs

The args for a ParamSpec object.

Given a ParamSpec object P, P.args is an instance of ParamSpecArgs.

ParamSpecArgs objects have a reference back to their ParamSpec::

    >>> P = ParamSpec("P")
    >>> P.args.__origin__ is P
    True

This type is meant for runtime introspection and has no special meaning
to static type checkers.

### ParamSpecKwargs

The kwargs for a ParamSpec object.

Given a ParamSpec object P, P.kwargs is an instance of ParamSpecKwargs.

ParamSpecKwargs objects have a reference back to their ParamSpec::

    >>> P = ParamSpec("P")
    >>> P.kwargs.__origin__ is P
    True

This type is meant for runtime introspection and has no special meaning
to static type checkers.

### Protocol

Base class for protocol classes.

Protocol classes are defined as::

    class Proto(Protocol):
        def meth(self) -> int:
            ...

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing).

For example::

    class C:
        def meth(self) -> int:
            return 0

    def func(x: Proto) -> int:
        return x.meth()

    func(C())  # Passes static type check

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as::

    class GenProto[T](Protocol):
        def meth(self) -> T:
            ...

### Request

A base class for incoming HTTP connections, that is used to provide
any functionality that is common to both `Request` and `WebSocket`.

#### Methods

##### `body(self) -> 'bytes'`



Type annotations:
```python
return: bytes
```

##### `close(self) -> 'None'`



Type annotations:
```python
return: None
```

##### `form(self, *, max_files: 'int | float' = 1000, max_fields: 'int | float' = 1000) -> 'AwaitableOrContextManager[FormData]'`



Type annotations:
```python
max_files: int | float
max_fields: int | float
return: AwaitableOrContextManager[FormData]
```

##### `get(self, key, default=None)`

D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.

##### `is_disconnected(self) -> 'bool'`



Type annotations:
```python
return: bool
```

##### `items(self)`

D.items() -> a set-like object providing a view on D's items

##### `json(self) -> 'typing.Any'`



Type annotations:
```python
return: typing.Any
```

##### `keys(self)`

D.keys() -> a set-like object providing a view on D's keys

##### `send_push_promise(self, path: 'str') -> 'None'`



Type annotations:
```python
path: str
return: None
```

##### `stream(self) -> 'typing.AsyncGenerator[bytes, None]'`



Type annotations:
```python
return: typing.AsyncGenerator[bytes, None]
```

##### `url_for(self, name: 'str', /, **path_params: 'typing.Any') -> 'URL'`



Type annotations:
```python
name: str
path_params: typing.Any
return: URL
```

##### `values(self)`

D.values() -> an object providing a view on D's values

### Server



#### Methods

##### `add_endpoint(self, name, fn)`



##### `check_user_data(self, address)`



##### `extract_time(self, x)`



##### `forward(self, fn: str, request: starlette.requests.Request, catch_exception: bool = True) -> dict`



Type annotations:
```python
fn: <class 'str'>
request: <class 'starlette.requests.Request'>
catch_exception: <class 'bool'>
return: <class 'dict'>
```

##### `get_data(self, request: starlette.requests.Request)`



Type annotations:
```python
request: <class 'starlette.requests.Request'>
```

##### `get_headers(self, request: starlette.requests.Request)`



Type annotations:
```python
request: <class 'starlette.requests.Request'>
```

##### `get_logs(self, tail=100, mode='local')`



##### `history(self, user)`



##### `is_admin(self, address)`



##### `remove_user_data(self, address)`



##### `resolve_path(self, path)`



##### `set_functions(self, functions: Optional[List[str]], fn2cost=None, helper_functions=['info', 'metadata', 'schema', 'free', 'name', 'functions', 'key_address', 'crypto_type', 'fns', 'forward', 'rate_limit'], functions_attributes=['helper_functions', 'whitelist', 'whitelist_functions', 'endpoints', 'functions', 'fns', 'exposed_functions', 'server_functions', 'public_functions'], free=False)`



Type annotations:
```python
functions: typing.Optional[typing.List[str]]
```

##### `set_key(self, key, crypto_type)`



##### `set_network(self, network)`



##### `set_port(self, port: Optional[int] = None, port_attributes=['port', 'server_port'], ip=None)`



Type annotations:
```python
port: typing.Optional[int]
```

##### `set_user_path(self, users_path)`



##### `start_server(self, max_bytes=10485760, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])`



##### `sync(self, update=True, state_keys=['stake_from', 'stake_to'])`



##### `sync_loop(self, sync_loop_initial_sleep=10)`



##### `user2count(self)`



##### `user2fn2count(self)`



##### `user_call_count(self, user)`



##### `user_call_path2latency(self, address)`



##### `user_call_paths(self, address)`



##### `user_data(self, address, stream=False)`



##### `user_path(self, key_address)`



##### `user_path2time(self, address)`



##### `users(self)`



##### `verify(self, fn: str, data: dict, headers: dict) -> bool`



Type annotations:
```python
fn: <class 'str'>
data: <class 'dict'>
headers: <class 'dict'>
return: <class 'bool'>
```

### SupportsAbs

An ABC with one abstract method __abs__ that is covariant in its return type.

### SupportsBytes

An ABC with one abstract method __bytes__.

### SupportsComplex

An ABC with one abstract method __complex__.

### SupportsFloat

An ABC with one abstract method __float__.

### SupportsIndex

An ABC with one abstract method __index__.

### SupportsInt

An ABC with one abstract method __int__.

### SupportsRound

An ABC with one abstract method __round__ that is covariant in its return type.

### Text

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### TextIO

Typed version of the return of open() in text mode.

#### Methods

##### `close(self) -> None`



Type annotations:
```python
return: None
```

##### `fileno(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `flush(self) -> None`



Type annotations:
```python
return: None
```

##### `isatty(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `read(self, n: int = -1) -> ~AnyStr`



Type annotations:
```python
n: <class 'int'>
return: ~AnyStr
```

##### `readable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `readline(self, limit: int = -1) -> ~AnyStr`



Type annotations:
```python
limit: <class 'int'>
return: ~AnyStr
```

##### `readlines(self, hint: int = -1) -> List[~AnyStr]`



Type annotations:
```python
hint: <class 'int'>
return: typing.List[~AnyStr]
```

##### `seek(self, offset: int, whence: int = 0) -> int`



Type annotations:
```python
offset: <class 'int'>
whence: <class 'int'>
return: <class 'int'>
```

##### `seekable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `tell(self) -> int`



Type annotations:
```python
return: <class 'int'>
```

##### `truncate(self, size: int = None) -> int`



Type annotations:
```python
size: <class 'int'>
return: <class 'int'>
```

##### `writable(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `write(self, s: ~AnyStr) -> int`



Type annotations:
```python
s: ~AnyStr
return: <class 'int'>
```

##### `writelines(self, lines: List[~AnyStr]) -> None`



Type annotations:
```python
lines: typing.List[~AnyStr]
return: None
```

### TypeAliasType

Type alias.

Type aliases are created through the type statement::

    type Alias = int

In this example, Alias and int will be treated equivalently by static
type checkers.

At runtime, Alias is an instance of TypeAliasType. The __name__
attribute holds the name of the type alias. The value of the type alias
is stored in the __value__ attribute. It is evaluated lazily, so the
value is computed only if the attribute is accessed.

Type aliases can also be generic::

    type ListOrSet[T] = list[T] | set[T]

In this case, the type parameters of the alias are stored in the
__type_params__ attribute.

See PEP 695 for more information.

### TypeVar

Type variable.

The preferred way to construct a type variable is via the dedicated
syntax for generic functions, classes, and type aliases::

    class Sequence[T]:  # T is a TypeVar
        ...

This syntax can also be used to create bound and constrained type
variables::

    # S is a TypeVar bound to str
    class StrSequence[S: str]:
        ...

    # A is a TypeVar constrained to str or bytes
    class StrOrBytesSequence[A: (str, bytes)]:
        ...

However, if desired, reusable type variables can also be constructed
manually, like so::

   T = TypeVar('T')  # Can be anything
   S = TypeVar('S', bound=str)  # Can be any subtype of str
   A = TypeVar('A', str, bytes)  # Must be exactly str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they
are created through the type parameter syntax and when
``infer_variance=True`` is passed. Manually created type variables may
be explicitly marked covariant or contravariant by passing
``covariant=True`` or ``contravariant=True``. By default, manually
created type variables are invariant. See PEP 484 and PEP 695 for more
details.

### TypeVarTuple

Type variable tuple. A specialized form of type variable that enables
variadic generics.

The preferred way to construct a type variable tuple is via the
dedicated syntax for generic functions, classes, and type aliases,
where a single '*' indicates a type variable tuple::

    def move_first_element_to_last[T, *Ts](tup: tuple[T, *Ts]) -> tuple[*Ts, T]:
        return (*tup[1:], tup[0])

For compatibility with Python 3.11 and earlier, TypeVarTuple objects
can also be created as follows::

    Ts = TypeVarTuple('Ts')  # Can be given any name

Just as a TypeVar (type variable) is a placeholder for a single type,
a TypeVarTuple is a placeholder for an *arbitrary* number of types. For
example, if we define a generic class using a TypeVarTuple::

    class C[*Ts]: ...

Then we can parameterize that class with an arbitrary number of type
arguments::

    C[int]       # Fine
    C[int, str]  # Also fine
    C[()]        # Even this is fine

For more details, see PEP 646.

Note that only TypeVarTuples defined in the global scope can be
pickled.

## Functions

### `NamedTuple(typename, fields=None, /, **kwargs)`

Typed version of namedtuple.

Usage::

    class Employee(NamedTuple):
        name: str
        id: int

This is equivalent to::

    Employee = collections.namedtuple('Employee', ['name', 'id'])

The resulting class has an extra __annotations__ attribute, giving a
dict that maps field names to types.  (The field names are also in
the _fields attribute, which is part of the namedtuple API.)
An alternative equivalent functional syntax is also accepted::

    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

### `TypedDict(typename, fields=None, /, *, total=True, **kwargs)`

A simple typed namespace. At runtime it is equivalent to a plain dict.

TypedDict creates a dictionary type such that a type checker will expect all
instances to have a certain set of keys, where each key is
associated with a value of a consistent type. This expectation
is not checked at runtime.

Usage::

    >>> class Point2D(TypedDict):
    ...     x: int
    ...     y: int
    ...     label: str
    ...
    >>> a: Point2D = {'x': 1, 'y': 2, 'label': 'good'}  # OK
    >>> b: Point2D = {'z': 3, 'label': 'bad'}           # Fails type check
    >>> Point2D(x=1, y=2, label='first') == dict(x=1, y=2, label='first')
    True

The type info can be accessed via the Point2D.__annotations__ dict, and
the Point2D.__required_keys__ and Point2D.__optional_keys__ frozensets.
TypedDict supports an additional equivalent form::

    Point2D = TypedDict('Point2D', {'x': int, 'y': int, 'label': str})

By default, all keys must be present in a TypedDict. It is possible
to override this by specifying totality::

    class Point2D(TypedDict, total=False):
        x: int
        y: int

This means that a Point2D TypedDict can have any of the keys omitted. A type
checker is only expected to support a literal False or True as the value of
the total argument. True is the default, and makes all items defined in the
class body be required.

The Required and NotRequired special forms can also be used to mark
individual keys as being required or not required::

    class Point2D(TypedDict):
        x: int               # the "x" key must always be present (Required is the default)
        y: NotRequired[int]  # the "y" key can be omitted

See PEP 655 for more details on Required and NotRequired.

### `assert_never(arg: Never, /) -> Never`

Statically assert that a line of code is unreachable.

Example::

    def int_or_str(arg: int | str) -> None:
        match arg:
            case int():
                print("It's an int")
            case str():
                print("It's a str")
            case _:
                assert_never(arg)

If a type checker finds that a call to assert_never() is
reachable, it will emit an error.

At runtime, this throws an exception when called.

Type annotations:
```python
arg: typing.Never
return: typing.Never
```

### `assert_type(val, typ, /)`

Ask a static type checker to confirm that the value is of the given type.

At runtime this does nothing: it returns the first argument unchanged with no
checks or side effects, no matter the actual type of the argument.

When a static type checker encounters a call to assert_type(), it
emits an error if the value is not of the specified type::

    def greet(name: str) -> None:
        assert_type(name, str)  # OK
        assert_type(name, int)  # type checker error

### `cast(typ, val)`

Cast a value to a type.

This returns the value unchanged.  To the type checker this
signals that the return value has the designated type, but at
runtime we intentionally don't check anything (we want this
to be as fast as possible).

### `clear_overloads()`

Clear all overloads in the registry.

### `dataclass_transform(*, eq_default: bool = True, order_default: bool = False, kw_only_default: bool = False, frozen_default: bool = False, field_specifiers: tuple[typing.Union[type[typing.Any], typing.Callable[..., typing.Any]], ...] = (), **kwargs: Any) -> <class '_IdentityCallable'>`

Decorator to mark an object as providing dataclass-like behaviour.

The decorator can be applied to a function, class, or metaclass.

Example usage with a decorator function::

    @dataclass_transform()
    def create_model[T](cls: type[T]) -> type[T]:
        ...
        return cls

    @create_model
    class CustomerModel:
        id: int
        name: str

On a base class::

    @dataclass_transform()
    class ModelBase: ...

    class CustomerModel(ModelBase):
        id: int
        name: str

On a metaclass::

    @dataclass_transform()
    class ModelMeta(type): ...

    class ModelBase(metaclass=ModelMeta): ...

    class CustomerModel(ModelBase):
        id: int
        name: str

The ``CustomerModel`` classes defined above will
be treated by type checkers similarly to classes created with
``@dataclasses.dataclass``.
For example, type checkers will assume these classes have
``__init__`` methods that accept ``id`` and ``name``.

The arguments to this decorator can be used to customize this behavior:
- ``eq_default`` indicates whether the ``eq`` parameter is assumed to be
    ``True`` or ``False`` if it is omitted by the caller.
- ``order_default`` indicates whether the ``order`` parameter is
    assumed to be True or False if it is omitted by the caller.
- ``kw_only_default`` indicates whether the ``kw_only`` parameter is
    assumed to be True or False if it is omitted by the caller.
- ``frozen_default`` indicates whether the ``frozen`` parameter is
    assumed to be True or False if it is omitted by the caller.
- ``field_specifiers`` specifies a static list of supported classes
    or functions that describe fields, similar to ``dataclasses.field()``.
- Arbitrary other keyword arguments are accepted in order to allow for
    possible future extensions.

At runtime, this decorator records its arguments in the
``__dataclass_transform__`` attribute on the decorated object.
It has no other runtime effect.

See PEP 681 for more details.

Type annotations:
```python
eq_default: <class 'bool'>
order_default: <class 'bool'>
kw_only_default: <class 'bool'>
frozen_default: <class 'bool'>
field_specifiers: tuple[typing.Union[type[typing.Any], typing.Callable[..., typing.Any]], ...]
kwargs: typing.Any
return: <class 'typing._IdentityCallable'>
```

### `final(f)`

Decorator to indicate final methods and final classes.

Use this decorator to indicate to type checkers that the decorated
method cannot be overridden, and decorated class cannot be subclassed.

For example::

    class Base:
        @final
        def done(self) -> None:
            ...
    class Sub(Base):
        def done(self) -> None:  # Error reported by type checker
            ...

    @final
    class Leaf:
        ...
    class Other(Leaf):  # Error reported by type checker
        ...

There is no runtime checking of these properties. The decorator
attempts to set the ``__final__`` attribute to ``True`` on the decorated
object to allow runtime introspection.

### `get_args(tp)`

Get type arguments with all substitutions performed.

For unions, basic simplifications used by Union constructor are performed.

Examples::

    >>> T = TypeVar('T')
    >>> assert get_args(Dict[str, int]) == (str, int)
    >>> assert get_args(int) == ()
    >>> assert get_args(Union[int, Union[T, int], str][int]) == (int, str)
    >>> assert get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
    >>> assert get_args(Callable[[], T][int]) == ([], int)

### `get_origin(tp)`

Get the unsubscripted version of a type.

This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar,
Annotated, and others. Return None for unsupported types.

Examples::

    >>> P = ParamSpec('P')
    >>> assert get_origin(Literal[42]) is Literal
    >>> assert get_origin(int) is None
    >>> assert get_origin(ClassVar[int]) is ClassVar
    >>> assert get_origin(Generic) is Generic
    >>> assert get_origin(Generic[T]) is Generic
    >>> assert get_origin(Union[T, int]) is Union
    >>> assert get_origin(List[Tuple[T, T]][int]) is list
    >>> assert get_origin(P.args) is P

### `get_overloads(func)`

Return all defined overloads for *func* as a sequence.

### `get_type_hints(obj, globalns=None, localns=None, include_extras=False)`

Return type hints for an object.

This is often the same as obj.__annotations__, but it handles
forward references encoded as string literals and recursively replaces all
'Annotated[T, ...]' with 'T' (unless 'include_extras=True').

The argument may be a module, class, method, or function. The annotations
are returned as a dictionary. For classes, annotations include also
inherited members.

TypeError is raised if the argument is not of a type that can contain
annotations, and an empty dictionary is returned if no annotations are
present.

BEWARE -- the behavior of globalns and localns is counterintuitive
(unless you are familiar with how eval() and exec() work).  The
search order is locals first, then globals.

- If no dict arguments are passed, an attempt is made to use the
  globals from obj (or the respective module's globals for classes),
  and these are also used as the locals.  If the object does not appear
  to have globals, an empty dictionary is used.  For classes, the search
  order is globals first then locals.

- If one dict argument is passed, it is used for both globals and
  locals.

- If two dict arguments are passed, they specify globals and
  locals, respectively.

### `is_typeddict(tp)`

Check if an annotation is a TypedDict class.

For example::

    >>> from typing import TypedDict
    >>> class Film(TypedDict):
    ...     title: str
    ...     year: int
    ...
    >>> is_typeddict(Film)
    True
    >>> is_typeddict(dict)
    False

### `no_type_check(arg)`

Decorator to indicate that annotations are not type hints.

The argument must be a class or function; if it is a class, it
applies recursively to all methods and classes defined in that class
(but not to methods defined in its superclasses or subclasses).

This mutates the function(s) or class(es) in place.

### `no_type_check_decorator(decorator)`

Decorator to give another decorator the @no_type_check effect.

This wraps the decorator with something that wraps the decorated
function in @no_type_check.

### `overload(func)`

Decorator for overloaded functions/methods.

In a stub file, place two or more stub definitions for the same
function in a row, each decorated with @overload.

For example::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...

In a non-stub file (i.e. a regular .py file), do the same but
follow it with an implementation.  The implementation should *not*
be decorated with @overload::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...
    def utf8(value):
        ...  # implementation goes here

The overloads for a function can be retrieved at runtime using the
get_overloads() function.

### `override(method: F, /) -> F`

Indicate that a method is intended to override a method in a base class.

Usage::

    class Base:
        def method(self) -> None:
            pass

    class Child(Base):
        @override
        def method(self) -> None:
            super().method()

When this decorator is applied to a method, the type checker will
validate that it overrides a method or attribute with the same name on a
base class.  This helps prevent bugs that may occur when a base class is
changed without an equivalent change to a child class.

There is no runtime checking of this property. The decorator attempts to
set the ``__override__`` attribute to ``True`` on the decorated object to
allow runtime introspection.

See PEP 698 for details.

Type annotations:
```python
method: F
return: F
```

### `reveal_type(obj: T, /) -> T`

Ask a static type checker to reveal the inferred type of an expression.

When a static type checker encounters a call to ``reveal_type()``,
it will emit the inferred type of the argument::

    x: int = 1
    reveal_type(x)

Running a static type checker (e.g., mypy) on this example
will produce output similar to 'Revealed type is "builtins.int"'.

At runtime, the function prints the runtime type of the
argument and returns the argument unchanged.

Type annotations:
```python
obj: T
return: T
```

### `runtime_checkable(cls)`

Mark a protocol class as a runtime protocol.

Such protocol can be used with isinstance() and issubclass().
Raise TypeError if applied to a non-protocol class.
This allows a simple-minded structural check very similar to
one trick ponies in collections.abc such as Iterable.

For example::

    @runtime_checkable
    class Closable(Protocol):
        def close(self): ...

    assert isinstance(open('/some/file'), Closable)

Warning: this will check only the presence of the required methods,
not their type signatures!

