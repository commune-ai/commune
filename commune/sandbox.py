import commune as c

x = {}
c.dict_put(x, 'brother.bro', ['sister'])
print(c.dict_get(x, 'brother.bro.0'))


print(x)