# agent



Source: `commune/modules/agent/agent.py`

## Classes

### Agent



#### Methods

##### `ask(self, *text, **kwargs)`



##### `edit(self, *args, file='./', **kwargs)`



##### `exe(self, *text, path='./', **kwargs)`



##### `forward(self, text='whats 2+2?', model='anthropic/claude-3.5-sonnet', temperature=0.5, max_tokens=1000000, stream=True)`



##### `generate(self, text='whats 2+2?', model='anthropic/claude-3.5-sonnet', temperature=0.5, max_tokens=1000000, stream=True)`



##### `models(self)`



##### `process_response(self, response)`



##### `process_text(self, text, threshold=1000)`



##### `reduce(self, text, max_chars=10000, timeout=40, max_age=30, model='openai/o1-mini')`



##### `score(self, module: str, **kwargs)`



Type annotations:
```python
module: <class 'str'>
```

