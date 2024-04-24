import commune as c

class MyModule(c.Module):
    whitelist = ["my_function"]
    def my_function(self):
        print("Hello, World!")
