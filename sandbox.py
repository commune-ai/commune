import commune as c
model = c.connect('model.openai', network='local')
c.print(model.forward('hello world'))
c.print(model.fns())