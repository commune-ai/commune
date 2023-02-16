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