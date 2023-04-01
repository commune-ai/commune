import commune

key = commune.key('Alice')
print(key.verify(key.sign('Hello, world!', return_dict=True)))