import commune



module = commune.connect('AgentExecutor')

print(module(fn='run', args= ['What is the name of the president of the United States?']))
