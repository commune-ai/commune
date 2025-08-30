from threading import Lock
from types import TracebackType
from typing import ContextManager, Generic, TypeVar

T = TypeVar("T")


class MutexBox(Generic[T], ContextManager[T]):
    _mutex: Lock
    _value: T

    def __init__(self, value: T):
        self._mutex = Lock()
        self._value = value

    def __enter__(self) -> T:
        self._mutex.acquire()
        return self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self._mutex.release()
