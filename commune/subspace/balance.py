# The MIT License (MIT)
# Copyright © 2021 Yuma nano

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from typing import Union


class Balance:
    """
    Represents the bittensor balance of the wallet, stored as nano (int)
    The Balance object is immutable, and can be used as a number or as a string
    Can only guarantee that the balance is accurate to 9 decimal places (token)

    Note: In operations between Balance and int/float, the other value is assumed to be in nano
    """

    unit: str = "\u03C4" # This is the token unit
    nano_unit: str = "\u03C1" # This is the nano unit
    nano: int
    token: float

    def __init__(self, balance: Union[int, float]):
        if isinstance(balance, int):
            self.nano = balance
        elif isinstance(balance, float):
            # Assume token value for the float
            self.nano = int(balance * pow(10, 9))
        else:
            raise TypeError("balance must be an int (nano) or a float (token)")

    @property
    def token(self):
        return self.nano / pow(10, 9)

    def __int__(self):
        return self.nano

    def __float__(self):
        return self.token

    def __str__(self):
        return f"{self.unit}{float(self.token):,.9f}"

    def __rich__(self):
        return "[green]{}[/green][green]{}[/green][green].[/green][dim green]{}[/dim green]".format(
            self.unit,
            format(float(self.token), "f").split(".")[0],
            format(float(self.token), "f").split(".")[1],
        )

    def __str_nano__(self):
        return f"{self.nano_unit}{int(self.nano)}"

    def __rich_nano__(self):
        return f"[green]{self.nano_unit}{int(self.nano)}[/green]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Union[int, float, "Balance"]):
        if other is None:
            return False
            
        if hasattr(other, "nano"):
            return self.nano == other.nano
        else:
            try:
                # Attempt to cast to int from nano
                other_nano = int(other)
                return self.nano == other_nano
            except (TypeError, ValueError):
                raise NotImplementedError("Unsupported type")

    def __ne__(self, other: Union[int, float, "Balance"]):
        return not self == other

    def __gt__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return self.nano > other.nano
        else:
            try:
                # Attempt to cast to int from nano
                other_nano = int(other)
                return self.nano > other_nano
            except ValueError:
                raise NotImplementedError("Unsupported type")

    def __lt__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return self.nano < other.nano
        else:
            try:
                # Attempt to cast to int from nano
                other_nano = int(other)
                return self.nano < other_nano
            except ValueError:
                raise NotImplementedError("Unsupported type")

    def __le__(self, other: Union[int, float, "Balance"]):
        try:
            return self < other or self == other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __ge__(self, other: Union[int, float, "Balance"]):
        try:
            return self > other or self == other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __add__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(self.nano + other.nano))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(self.nano + other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __radd__(self, other: Union[int, float, "Balance"]):
        try:
            return self + other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __sub__(self, other: Union[int, float, "Balance"]):
        try:
            return self + -other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __rsub__(self, other: Union[int, float, "Balance"]):
        try:
            return -self + other
        except (TypeError):
            raise NotImplementedError("Unsupported type")

    def __mul__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(self.nano * other.nano))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(self.nano * other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rmul__(self, other: Union[int, float, "Balance"]):
        return self * other

    def __truediv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(self.nano / other.nano))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(self.nano / other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rtruediv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(other.nano / self.nano))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(other / self.nano))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __floordiv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(self.token // other.token))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(self.nano // other))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __rfloordiv__(self, other: Union[int, float, "Balance"]):
        if hasattr(other, "nano"):
            return Balance.from_nano(int(other.nano // self.nano))
        else:
            try:
                # Attempt to cast to int from nano
                return Balance.from_nano(int(other // self.nano))
            except (ValueError, TypeError):
                raise NotImplementedError("Unsupported type")

    def __int__(self) -> int:
        return self.nano

    def __float__(self) -> float:
        return self.token

    def __nonzero__(self) -> bool:
        return bool(self.nano)

    def __neg__(self):
        return Balance.from_nano(-self.nano)

    def __pos__(self):
        return Balance.from_nano(self.nano)

    def __abs__(self):
        return Balance.from_nano(abs(self.nano))

    @staticmethod
    def from_float(amount: float):
        """Given token (float), return Balance object with nano(int) and token(float), where nano = int(token*pow(10,9))"""
        nano = int(amount * pow(10, 9))
        return Balance(nano)

    @staticmethod
    def from_token(amount: float):
        """Given token (float), return Balance object with nano(int) and token(float), where nano = int(token*pow(10,9))"""
        nano = int(amount * pow(10, 9))
        return Balance(nano)

    @staticmethod
    def from_nano(amount: int):
        """Given nano (int), return Balance object with nano(int) and token(float), where nano = int(token*pow(10,9))"""
        return Balance(amount)
