

## Model.VectorStore Module

#### Bash
```bash
c model.vectorstore serve tag=10 api_key=.....
```
#### Python
```python
c.serve('model.vectorstore', tag=10, api_key=...)
```

To register the module

#### Bash
```bash
c model.vectorstore register tag=10 api_key=...
```

#### Python
```python
c.register('model.vectorstore', tag=10, api_key='....')
```

To run the module

#### Bash
```bash
c model.vectorstore search api_key=... path=... query=...
```

#### Python
```python
c model.vectorstore search api_key="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" path="state_of_the_union.txt" query="What did the president say about Ketanji"
```

Add embedding sentence
#### Bash
```bash
c model.vectorstore add_sentence sentence="fam is owner of commune."
```

Get sentence by prompting
#### Bash
```bash
c model.vectorstore prompt query="Who is fam?"
```
