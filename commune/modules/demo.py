import commune as c 

class Demo(c.Module):
    
    @c.endpoint
    def forward22(self, x=1, y=1):
        return  x + y
    
    def test(self):
        assert self() == 2
        assert self(2, 3) == 5
        assert self(3, 4) == 7