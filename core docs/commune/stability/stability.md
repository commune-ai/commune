# StabilityAI Module

This module provides an interface to interact with Stability AI's API.

The `StabilityAI` class provides functions for requesting a number of AI features, including text-to-image, image-to-image, and image masking.

## Initialization

The class requires an API Key and Engine ID on initialization. It also allows you to specify the API host. 

```python
stability_ai = StabilityAI(engine_id="engine_id", api_host="api_host", api_key="api_key")
```

**Methods**:  

1. `account`: Retrieves the account details of the user.
2. `image2image`: Takes an initial image and a text prompt and calls the Stability AI API to generate a new image based on the prompt.
3. `text2image`: Takes a text prompt and calls the Stability AI API to generate an image based on the prompt.
4. `upscale_image`: Upscales an image by a certain width.
5. `image_masking`: Uses the Stability AI API to mask parts of an image based on a text prompt.
6. `text2video`: Creates a video from a text prompt.
7. `test`: Tests the StabilityAI class functionality.
8. `test_image2image`: Tests the image2image method.
9. `test_text2image`: Tests the text2image method.

Private Methods:

1. `_make_request`: Sends a request to the set API host, handling Authorization.
2. `_process_response`: Processes the response from the API call and returns an array of images.

### How to use

Please refer to the function docstrings for specific details on function usage. Here's a basic example:

```python
stability_ai = StabilityAI(api_key="your_api_key_here")
account_info = stability_ai.account() # Get your account's details
```