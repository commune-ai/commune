# Pytorch GANs Package

This package contains helper functions and classes for building Generative Adversarial Networks (GANs), specifically those involving Convolutional Neural Networks (CNNs), using Pytorch. It includes building blocks for state-of-the-art GAN architectures like BigGAN and StyleGAN. 

## Features
- Wrapper functions for common Pytorch modules (Conv2D, ConvTranspose2D, Embedding, Linear) with spectral normalization applied. Spectral normalization is a weight regularization technique for stabilizing the training of GANs.
- NormLayer function for returning either a Group Normalization or a Batch Normalization layer.
- Activations function classes (GLU, Swish).
- InitLayer for the initial transformation of the input noise vector in the GAN's generator.
- UpBlock (for U-net style generator) and DownBlock (for discriminator) classes for performing upsampling and downsampling respectively. They help in feature map dimension adjustment.
- SEBlock for dynamic channel-wise feature recalibration.
- SeparableConv2d class for implementing separable convolutions, enhancing efficiency.
- Classes for constructing a context-sensitive module (CSM): ResidualConvUnit and FeatureFusionBlock.
- NoiseInjection and CCBN classes for using specific tools found in StyleGAN and BigGAN architectures.
- Interpolation functionality wrapped in a class (Interpolate).

## Usage
To use this package, you need to have Pytorch installed in your environment. Import the modules you wish to use from it in your Python script. You can use them as building blocks to construct your GAN models.

## Examples
Examples of usage can be found in the test files provided in the package.

## Contributing
Contributions to improve this package are always welcome. You can create a fork, make your changes, and then issue a pull request. Please make sure to follow the existing style conventions and structure.
