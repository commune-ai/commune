import commune



module = commune.connect('ReactAgentModule')

print(module(fn='run', args= ['What is the name of the president of the United States?']))
