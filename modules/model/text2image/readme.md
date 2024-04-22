# Convert Text to Image Module - README

This module is designed for converting user prompt to call. It offers a `generate` function that users can convert the text with customized parameters.
Here is the detailed explanation of the `generate` function. Also I added docstring so that you can easily commit.

## Here's a detailed explanation of parameters for convert:

-`Prompt`:           Prompt to generate image.
-`SaveFileName`:     Filename to save the results.


### Usage Example:

`c model.text2image generate Prompt="a cat sitting on the chair" SaveFileName="cat"`

`c model.text2image gradio`

You can find diffusion models in the huggingface hub.
https://huggingface.co/models?other=diffusers:StableDiffusionPipeline&sort=trending
