

## Model.VectorStore Module

#### Add OpenAI API Key
Save your own OpenAI API KEY in .env file.

```bash
OPENAI_API_KEY=Your OpenAI API KEY
```

#### Install dependencies

```bash
pip install -r requirements.txt
```
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
c model.vectorstore search path=... query=...
```

#### Python
```python

c model.vectorstore search path="state_of_the_union.txt" query="What did the president say about Ketanji"
```

Add embedding sentence
#### Bash
```bash
c model.vectorstore add_sentence sentence="fam is owner of commune."
```

Add embedding from file
#### Bash
```bash
c model.vectorstore add_from_file path=...
```

Add embedding sentence
#### Bash
```bash
c model.vectorstore add_from_url url=...
```

Get sentence by prompting
#### Bash
```bash
c model.vectorstore prompt query="Who is fam?"
```

Launch gradio for testing
#### Bash
```bash
c model.vectorstore gradio
```

#### Note
Once you install chromadb package, it'd not be able to launch gradio.
At that time, you need to modify some code in the nest_asyncio.py, line 29.

Replace
```bash
loop = asyncio.get_event_loop()
```
with
```bash
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
```
