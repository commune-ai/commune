import commune as c

class Sandbox(c.Module):
    def store_something(self, x=1):
        self.put('something', x)

    def get_something(self):
        return self.get('something')
