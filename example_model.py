import commune as c

# after installing commune call me
# c example_model/predict a=10 b=10 
# c example_model/predict 10 10

class Example(c.Module):
    def predict(self, a: int = 10, b: int = 10):
        return a + b
    