import commune as c
from typing import *
import torch

class MatMul(c.Module):
    device = 'mps'

    def forward(self, x = None, y = None, n=20):
        if x is None or y is None:
            x =  torch.rand(n, n, device=self.device) 
            y = torch.rand(n, n, device=self.device)
        return torch.matmul(x, y)
    
    def test(self, n=3):
        t0 = c.time()
        x = torch.rand(n, n, device=self.device)
        y = torch.rand(n, n, device=self.device)
        t1 = c.time()
        result = self.forward(x, y, n)
        result_shape = list(result.shape)
        num_ops = n * n * n
        ops_per_sec = num_ops / (t1 - t0)
        # garbage collect
        del x, y, result

        return {"time": t1 - t0, "result_shape": result_shape, 'n': n, 'num_ops': num_ops , 'ops_per_sec': ops_per_sec}
    

    def test_plot(self,  num_points=80,  function=lambda x: x**2):
        results = []
        for i in range(num_points):
            result = self.test(n=function(i))
            result['i'] = i
            c.print(result)
            results.append(result)

        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(results)
        fig = px.line(df, x="n", y="ops_per_sec")
        fig.show()
        return df