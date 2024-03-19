# Convert Video to Text Module - README

This module is designed for generating text description of short video. It offers a `generate` function that users can convert the text with customized parameters.
Here is the detailed explanation of the `generate` function. Also I added docstring so that you can easily commit.

## Setting a Replicate Key

To set a [Replicate](https://replicate.com) key, use the following command:

```bash
c model.video2text add_api_key YOUR_API_KEY
```

Once this is done, you won't need to specify the key every time you start your module. The key will be saved in the module's folder at `.commune/model.video2text/api_keys.json`.

To remove an API key, use:

```bash
c model.video2text rm_api_key YOUR_API_KEY
```

To list all API keys:

```bash
c model.video2text api_keys
```

## Here's a detailed explanation of parameters for convert:

- `api_key`; default: None, # api key
- `image2text_model`; default = 'gfodor/instructblip', # image2text model
- `video`; default: "test_video.mp4", # video file path
- `interval`; default: 5, # time interval
- `summarize`; default: True, # summarize the desc
- `summarize_model`;    default: 'replicate/flan-t5-xl', # summarize model
- `summarize_max_len`;  default: 50, # Maximum number of tokens to generate. A word is generally 2-3 tokens
- `summarize_top_p`;    default: 0.95, # When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens
- `summarize_temp`;     defaultt: 0.7 # Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.

## Usage Example:

Here's an example of how to use the command:

`c model.video2text describe image2text_model="gfodor/instructblip" video="test_video.mp4" interval=5 summarize=True`

## Use gradio for test:

You can launch gradio interface using this command:

`c model.video2text gradio