
from typing import Union
import commune as c

class Balance:
    decimals = 9
    def __init__(self, joules: float):
        self.joules = joules
        self.nano = self.to_nano(joules)
        
    @classmethod
    def from_nano(cls, amount: int):
        """Given nano (int), return Balance object with nano(int) and joules(float), where nano = int(joules*pow(10,9))"""
        return amount  * (10 ** cls.decimals)

    @classmethod
    def to_nano(cls, amount: int):
        """Given nano (int), return Balance object with nano(int) and joules(float), where nano = int(joules*pow(10,9))"""
        return amount // (10 ** cls.decimals)
