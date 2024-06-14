### LiteLLM: A Lightweight LLM Module Documentation

`LiteLLM` is a simple module built on top of the `commune` framework that provides a straightforward way to communicate with the Language Learning Model (LLM) using the `litellm` library. Here's a comprehensive guide to use `LiteLLM`.

#### Installation

Before using `LiteLLM`, ensure the required libraries are installed. You can install `commune` and `litellm` using pip:

```bash
pip install commune litellm
```

#### Setting up `LiteLLM`

```python
import commune as c
from LiteLLM_module import LiteLLM

# Initialize LiteLLM with your API key
llm = LiteLLM(api_key="YOUR_API_KEY")
```

Note: If your API key is stored in an environment variable named "OPENAI_API_KEY", you don't need to pass it explicitly during initialization.

#### Making a Call

Use the `call` method to communicate with the model:

```python
response = llm.call(text="Hello, world!")
print(response)
```

#### Serving as an API

To serve `LiteLLM` as an API, use the following command:

```bash
c model.litellm serve [OPTIONS]
```

Where `[OPTIONS]` can be any arguments or configurations you wish to pass.

#### Additional Information

- **set_api(api_key: str)**: Set the API key for the LLM.
- **resolve_api_key(api_key: str)**: Resolves the given API key or falls back to the internally stored one.
- **call(text: str, messages: List[Dict[str, str]], model: str, api_key: str, **kwargs)**: The main function to make a call to the LLM. The `messages` parameter can be a list of previous interactions for context.

#### Example with Previous Interactions:

```python
previous_messages = [
    {'role': 'user', 'content': 'Hello!'},
    {'role': 'bot', 'content': 'Hi there!'},
]

response = llm.call(text="How are you?", messages=previous_messages)
print(response)
```

#### Class Methods

- **install()**: Installs the required `litellm` library if not already present.

### Summary

`LiteLLM` provides a convenient and lightweight way to interact with the LLM using `litellm` while maintaining a structure suitable for the `commune` framework. Whether you're using it standalone or serving it as an API, it's designed to be intuitive and easy to use.