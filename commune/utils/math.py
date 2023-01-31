
def round_sig(x, sig=6, small_value=1.0e-9):
    import math
    """
    Rounds x to the number of {sig} digits
    :param x:
    :param sig: signifant digit
    :param small_value: smallest possible value
    :return:
    """
    return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)



        
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value * count
        self.count = count

    def update(self, value, count=1):
        self.total_value += value * count
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

