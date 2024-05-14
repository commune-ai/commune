import commune as c
namespace = c.namespace()

for name, address in namespace.items():
    c.print(c.update_module(name, address=address))