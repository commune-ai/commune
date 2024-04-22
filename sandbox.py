import commune as c

class MyModule(c.Module):
    whitelist = ["my_function"]

    def my_function(self):
        print("Hello, World!")


c.print(MyModule().path2objectpath())
