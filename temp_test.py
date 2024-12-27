import commune as c

# Create and serve a module
class MyModule(c.Module):
    def process(self, data):
        return f"Processed: {data}"

# Start local server
c.serve('my_module', network='local', port=8000)

# Connect from another process
client = c.connect('my_module')
result = client.process("test data")