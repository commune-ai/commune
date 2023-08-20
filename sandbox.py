
import commune as c


dataset = c.connect('')

model = c.connect('model.llamaint4')

prompt = '''
Has the following text been tampered with yes (1) or no (0)?

TEXT

RESPONSE -> JSON({answer: float})
{'answer': '''
output = model.genertate('hello world', max_output_tokens=128)