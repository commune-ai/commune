# DreamBooth

This script provides an implementation for training `DreamBooth`, a system that generates an 3D model from text prompts. The implementation includes a few critical components including a text encoder for understanding the text prompts, a UNet model that uses the encoded features of the prompt to generate a 3D model, a training loop and model management functions. 

## Features

1. Load text encoder and UNet models from pre-trained checkpoints
2. A custom PyTorch Dataset which loads instance images and prompts to guide the model training
3. Training loop that includes support for mixed precision training, learning rate scheduling, and gradient accumulation
4. DiffusionPipeline that creates a pipeline using UNet and tokenizer for model generating
5. Option to save and load the training state for experimentation or transfer learning
6. Option to upload the model to Hugging Face hub

## Dependencies

- PyTorch for the underlying deep learning framework
- Accelerate for optimizing the training on multiple GPUs
- transformers for using and handling models and tokenizers
- diffusers for providing autoencoder, diffusion pipelines and other utilities
- xformers, if enabled
- PIL, torchvision for handling images
   and more...

Do ensure that these dependencies are installed properly. 

## Usage

Run the script in the terminal:

```shell
python dreambooth.py --option1 value1 --option2 value2
```

Replace `option1` and `option2` with appropriate arguments depending on your requirements. Do note that some arguments such as `pretrained_model_name_or_path` or `learning_rate` are mandatory, while others have default values if not specified.

The script also supports mixed-precision training and gradient accumulation for optimizing memory usage during training. Ensure that your hardware and software configuration supports these features before enabling them.

Refer to the script's argument parser (`parse_args()`) for a full list of arguments and their descriptions. 

## License

This project (DreamBooth) is licensed under the Apache 2.0 License, reproduced at the top of the script. Please review it for details about your rights and limitations under the license. The next lines are copyright information regarding the specific implementation provided in the script. 

## Disclaimer
The implementation may not fully represent the theoretical and experimental details provided in the original papers. Please use this script for academic or research purposes, and remember to cite the original authors if you use or repurpose parts of the implementation.