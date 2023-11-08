# Convert Text to Video Module - README

This module is designed for converting user prompt to video. It offers a `convert` function that users can convert the text with customized parameters.
Here is the detailed explanation of the `convert` function. Also I added docstring so that you can easily commit.

## Here's a detailed explanation of parameters for convert:

- `prompt`: A string that user wants to make a video.
- `steps`: This the parameter that present num_inferences_steps.
- `output`: This is output filename.


### Usage Example:

Here's an example of how to use the command:

`c model.text2video convert prompt="spiderman is surfing" steps=25 output="surfing"`
