import commune as c
model = c.connect('model.openai.free')
c.print(model.talk('hello'))