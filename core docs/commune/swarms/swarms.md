# Swarms Module with Commune Library

This is a simple script showing how to use Python's [Commune](https://pypi.org/project/commune/) library along with the [Request](https://docs.python-requests.org/en/master/) library to communicate with a SWARMs AI server.

The primary class is `Swarms`, which is a subclass of Commune's `Module`. This script demonstrates how to implement a `forward` method to facilitate communication with the SWARMs AI server.

In this script, the `Swarms` class is initialized with a server URL, and the `forward` method is used to send a HTTP POST request to SWARMs API for text/image completion.

## Using the `image_to_base64` Function

This function is used to convert an image to a Base64 string representation, which can be utilized to send images as part of an HTTP request.

```python
def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
```

## Initializing the `Swarms` Class

```python
class Swarms(c.Module):
    def __init__(self, url="https://api.swarms.world/v1/chat/completions"):
        self.url = url
```

The `Swarms` class makes a POST request to the specified URL using the function `forward`. If no URL is provided during the initialization, it defaults to `"https://api.swarms.world/v1/chat/completions"`.

## Making a POST Request with the `forward` Method

The `forward` method is used to make a POST request to the SWARMs API for text completion, given some parameters. These parameters include: text, image, max_tokens, temperature, model, role, top_p.

```python
def forward(self, 
            text="Describe what is in the image",
            image=None, 
            max_tokens=1024, 
            temperature=0.8,
            model="cogvlm-chat-17b",
            role = "user",
            top_p=0.9):
```

## Sending the Request

Once the adequate parameters are supplied, a request body is created and it is sent as a POST request to the SWARMs server. The response of the server is outputted as the return value of the `forward` method.

## Commune Library Dependency

Ensure that the Python environment has the Commune library. It can be installed via pip:

```bash
pip install commune
```