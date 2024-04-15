import commune as c
EPS = 1e-10
class Math(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def add(self, x:int = 1, y:int = 2) -> int:
        return x + y
    def subtract(self, x:int = 1, y:int = 2) -> int:
        return x - y
    def multiply(self, x:int = 1, y:int = 2) -> int:
        return x * y
    
    def divide(self, x:int = 1, y:int = 2) -> int:
        return x / (y + EPS)
    
    def median(self, entries: list) -> float:
        entries.sort()
        n = len(entries)
        if n % 2 == 1:
            return entries[n//2]
        return (entries[n//2 - 1] + entries[n//2]) / 2.0
    
    def mean(self, entries: list) -> float:
        return sum(entries) / len(entries)
    
    def mode(self, entries: list) -> float:
        count = {}
        for i in entries:
            count[i] = count.get(i, 0) + 1
        return max(count, key=count.get)
    