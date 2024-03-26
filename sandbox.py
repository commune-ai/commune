import commune as c


m = c.m()()
data = m.sign('hello', return_str=1)
c.print(m.verify(data))