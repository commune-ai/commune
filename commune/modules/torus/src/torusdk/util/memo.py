import time
from collections import OrderedDict
from collections.abc import MutableMapping
from threading import Lock
from typing import Callable, Generic, Iterator, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLDict(Generic[K, V], MutableMapping[K, V]):
    """
    A dictionary that expires its values after a given timeout.

    Time tracking is done with `time.monotonic_ns()`.

    Be careful, this doens't have and automatic cleanup of expired values yet.
    """

    ttl: int
    _values: OrderedDict[K, tuple[int, V]]
    _lock: Lock

    def __init__(
        self,
        ttl: int,
        _dict_type: type[OrderedDict[K, tuple[int, V]]] = OrderedDict,
    ):
        """
        Args:
            ttl: The timeout in seconds for the memoization.
        """
        self.ttl = ttl
        self._values = _dict_type()
        self._lock = Lock()

    @property
    def ttl_in_ns(self) -> int:
        return self.ttl * 1_000_000_000

    def __repr__(self) -> str:
        return f"<TTLDict@{id(self):#08x} ttl={self.ttl}>"

    def __is_expired(self, key: K) -> bool:
        expire_time, _ = self._values[key]
        return time.monotonic_ns() > expire_time

    def __remove_if_expired(self, key: K) -> bool:
        expired = self.__is_expired(key)
        if expired:
            del self._values[key]
        return expired

    def clean(self):
        with self._lock:
            for key in self._values.keys():
                removed = self.__remove_if_expired(key)
                if not removed:
                    # TODO: test cleanup optimization
                    break

    def __setitem__(self, key: K, value: V):
        with self._lock:
            expire_time = time.monotonic_ns() + self.ttl_in_ns
            self._values[key] = (expire_time, value)
            self._values.move_to_end(key)

    def __getitem__(self, key: K) -> V:
        with self._lock:
            self.__remove_if_expired(key)
            value = self._values[key][1]
            return value

    def __delitem__(self, key: K):
        with self._lock:
            del self._values[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            for key in self._values.keys():
                if not self.__remove_if_expired(key):
                    yield key

    def __len__(self) -> int:
        """
        Warning: this triggers a cleanup, and is O(n) in the number of items in
        the dict.
        """
        self.clean()

        # TODO: there is a race condition here.
        # Because we are not using RLock as we expect this crap to be used with async.
        # But I don't care. Be happy enough with with an "aproximate value".

        with self._lock:
            return len(self._values)

    def get_or_insert_lazy(self, key: K, fn: Callable[[], V]) -> V:
        """
        Gets the value for the given key, or inserts the value returned by the
        given function if the key is not present, returning it.
        """
        if key in self:
            return self[key]
        else:
            self[key] = fn()
            return self[key]


def __test():
    m: TTLDict[str, int] = TTLDict(1)

    m["a"] = 2

    print(m.get("a", default="missing"))
    print(m["a"])

    time.sleep(0.5)

    print(m.get("a", default="missing"))
    print(m["a"])

    time.sleep(1)

    print(m.get("a", default="missing"))
    try:
        print(m["a"])

    except KeyError:
        print("Key is not present :) yay")

    print(len(m))

    print()
    print()

    _counter = 0

    def generate():
        nonlocal _counter
        _counter += 1
        print(f"-> generated {_counter}")
        return _counter

    print("FIRST RUN")
    v = m.get_or_insert_lazy("a", generate)
    print(v)
    v = m.get_or_insert_lazy("a", generate)
    print(v)
    print()

    time.sleep(1.5)

    print("SECOND RUN")
    v = m.get_or_insert_lazy("a", generate)
    print(v)
    v = m.get_or_insert_lazy("a", generate)
    print(v)


if __name__ == "__main__":
    __test()
