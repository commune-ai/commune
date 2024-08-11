import commune as c 

class Demo(c.Module):
    
    @c.endpoint()
    def forward222(self, x=1, y=1):
        return  x + y + 1
    
