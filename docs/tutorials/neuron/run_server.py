import commune
model = commune.get_module('model.transformer')(model_name='gpt125m', tokenizer='gptj')
model.serve()