
from typing import Union, Dict, List, Tuple, Optional


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
        if self.count == 0:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)
    
    def to_dict(self):
        return self.__dict__()


    def from_dict(  self, 
                    d: Dict,
                    ):
        for key, value in d.items():
            assert hasattr(self, key), f'key {key} not in {self.__class__.__name__}'
            setattr(self, key, value)
        return self



        
class MovingWindowAverage:
    def __init__(self,value: Union[int, float] = None, window_size:int=100):
        self.set_window( value=value, window_size=window_size
        

    def set_window(self,value: Union[int, float] = None, window_size:int=100) -> List[Union[int, float]]:
        assert type(value) in [int, float], f'default_value must be int or float, got {type(default_value)}'
        self.window_size = window_size
        self.update(value)
        return self.window_values

    def update(self, *values):
        '''
        Update the moving window average with a new value.
        '''
        if hasattr(self, 'window_values') == False:
            self.window_values = []
            
        for value in values:
            self.window_values +=  [value]
            if len(self.window_values) > self.window_size:
                self.window_values = self.window_values[-self.window_size:]
                
        self.value = sum(self.window_values) / len(self.window_values)

    def __str__(self):
        return str(self.value)
    
    def to_dict(self):
        return self.__dict__()
    
    def from_dict(self, d: Dict):
        for key, value in d.items():
            assert hasattr(self, key), f'key {key} not in {self.__class__.__name__}'
            setattr(self, key, value)
        return self
    
    def to_json(self):
        return json.dumps(self.to_dict())
    
    def from_json(self, json_str:str):
        state_dict = json.loads(json_str)
        self.__dict__.update(state_dict)
        return state_dict

    def state_dict(self):
        return self.to_dict()

    @classmethod
    def test(cls):
        
        # testing constant value
        constant = 10
        self = cls(value=constant)
        
        for i in range(10):
            self.update(10)
            assert constant == self.value
            
        variable_value = 100
        window_size = 10
        self = cls(value=variable_value, window_size=window_size+1)
        for i in range(variable_value+1):
            self.update(i)
        print(self.value)
        assert self.value == (variable_value - window_size/2)
        print(self.window_values)
            