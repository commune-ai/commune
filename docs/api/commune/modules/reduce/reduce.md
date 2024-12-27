# reduce



Source: `commune/modules/reduce/reduce.py`

## Classes

### Reduce



#### Methods

##### `file2text(self, path)`



##### `files(self, path='./', query='the file that is the core of this folder', n=10, model='anthropic/claude-3.5-sonnet-20240620:beta')`



##### `forward(self, text, model='anthropic/claude-3.5-sonnet-20240620:beta', timeout=10, n=10)`



##### `modules(self, query='the filel that is the core of commune', model='anthropic/claude-3.5-sonnet-20240620:beta')`



##### `process_data(self, data)`



##### `query(self, options, query='most relevant modules', output_format='DICT(data:list[[idx:str, score:float]])', anchor='OUTPUT', n=10, model='anthropic/claude-3.5-sonnet-20240620:beta')`



##### `utils(self, query='confuse the gradients', model='anthropic/claude-3.5-sonnet-20240620:beta')`



