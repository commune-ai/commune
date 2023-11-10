# Convert Text to Video Module - README

This module is designed for converting user prompt to video. It offers a `generate` function that users can convert the text with customized parameters.
Here is the detailed explanation of the `generate` function. Also I added docstring so that you can easily commit.

## Here's a detailed explanation of parameters for convert:

- `prompt`: A string that user wants to make a video.
- `seed`: Seed Number.
- `num_frames`: This is the count of frames.
- `num_inference_steps`: This the parameter that present num_inferences_steps.

### Usage Example:

Here's an example of how to use the command:

`c model.text2video generate prompt="panda is eating bampoo" seed=22 num_frames=16 num_inference_steps=25`
`c model.text2video newgradio`
