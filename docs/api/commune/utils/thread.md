# thread



Source: `commune/utils/thread.py`

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

### `as_completed(futures: list, timeout: int = 10, **kwargs)`



Type annotations:
```python
futures: <class 'list'>
timeout: <class 'int'>
```

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

### `is_coroutine(future)`

is this a thread coroutine?

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

### `submit(fn, params=None, kwargs: dict = None, args: list = None, timeout: int = 40, return_future: bool = True, init_args: list = [], init_kwargs: dict = {}, executor=None, module: str = None, mode: str = 'thread', max_workers: int = 100)`



Type annotations:
```python
kwargs: <class 'dict'>
args: <class 'list'>
timeout: <class 'int'>
return_future: <class 'bool'>
init_args: <class 'list'>
init_kwargs: <class 'dict'>
module: <class 'str'>
mode: <class 'str'>
max_workers: <class 'int'>
```

### `thread(fn: Union[ForwardRef('callable'), str], args: list = None, kwargs: dict = None, daemon: bool = True, name=None, tag=None, start: bool = True, tag_seperator: str = '::', **extra_kwargs)`



Type annotations:
```python
fn: typing.Union[ForwardRef('callable'), str]
args: <class 'list'>
kwargs: <class 'dict'>
daemon: <class 'bool'>
start: <class 'bool'>
tag_seperator: <class 'str'>
```

### `threads(search: str = None)`



Type annotations:
```python
search: <class 'str'>
```

### `wait(futures: list, timeout: int = None, generator: bool = False, return_dict: bool = True) -> list`



Type annotations:
```python
futures: <class 'list'>
timeout: <class 'int'>
generator: <class 'bool'>
return_dict: <class 'bool'>
return: <class 'list'>
```

