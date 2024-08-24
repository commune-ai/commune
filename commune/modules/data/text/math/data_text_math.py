import commune as c
import random 
class DataTextMath(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)
        self.operations = {
            'add': '+',
            'subtract': '-',
            'multiply': '*',
            'divide': '/',
            'modulo': '%',
        }
    @staticmethod
    def random_value( bounds = [-1, 1]):
        output = 0
        while output == 0:
            output = random.random() * (bounds[1] - bounds[0]) + bounds[0]
        return output


    def sample(self, n = 2):
        op_chain = []
        y = self.random_value()
        for i in range(n):
           
            op = random.choice(list(self.operations.keys()))
            op_chain.append([op])
            # float between -1 and 1
            x = self.random_value()
            y = c.round(y, 3)
            x = c.round(x, 3)
            
            opstr = f'{y} {self.operations[op]} {x}'
            y = eval(opstr)

            op_chain[-1].append(opstr)

        sample = {
            'opperation_chain': op_chain,
            'answer': y
        }
        return sample

    @classmethod
    def test(cls, n=100):
        self = cls()
        t = c.time()
        for i in range(n):
            sample = self.sample()
            msg = {'samples_per_second': i / (c.time() - t)}
            c.print(msg, sample)
