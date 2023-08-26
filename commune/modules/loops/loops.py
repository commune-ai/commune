import commune as c

class Loops(c.Module):
    loops = {}
    @classmethod
    def add_loop(cls, module, fn, loop, period:int = 1):
        c.print(f"Adding loop {fn} to {module}", color='green')

            

        