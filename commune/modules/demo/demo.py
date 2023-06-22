import commune as c

class Demo(c.Module):
    
    def bro(self, x='fam'):
        return f'whadup {x}'
    
    def hey(self, x='fam'):
        return f'whadup {x}'
    

Demo.run(__name__)