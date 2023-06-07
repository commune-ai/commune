class Balance:
    decimals = 9
    @classmethod
    def from_token(cls, x: int, decimals: int = 9) -> float:
        return x / (10 ** decimals)
    def to_nano(self, x) -> int:
        return int(x * 10 ** decimals)