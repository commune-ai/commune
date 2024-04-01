# PyTorch FastGAN Projected Discriminator

This Python module defines a conditional GAN-based discriminator for the FastGAN architecture that uses a projection-based conditioning approach.

## Classes

- `SingleDisc`:  This class applies a series of convolutional layers and downsampling blocks to the input data. It assumes that the input data is an image tensor.
  
- `SingleDiscCond`: Class similar to the `SingleDisc`, but it also takes into account the given condition information to control the output of the GAN. It adds additional layers for conditioning on class data.

- `MultiScaleD`: This class uses several instances of the `SingleDisc` class for multi-scale conditional GAN. It takes in several features at different resolutions.
   
- `ProjectedDiscriminator`: This class feeds the given data through the feature network and the discriminator. It applies conditional augmentation if enabled and interpolates the image to a higher resolution.

## Key Components

- `nc`: the number of input channels
- `ndf`: the multiplication factor for the number of filters
- `start_sz`: the initial spatial size of the image
- `end_sz`: the final spatial size of the image
- `head`: if True, additional layers are added at the beginning
- `separable`: if True, uses depthwise separable convolutions
- `patch`: if True, uses patch-based discriminator
- `c_dim`: dimensionality of the conditional information
- `cmap_dim`: dimensionality of the conditional map
- `embedding_dim`: dimensionality of the embedded space

Note: Depthwise Separable Convolutions can reduce the number of parameters and computations in the model, thus making it more efficient. Patch-based discriminators consider patches of images independently, which might be useful for larger images, or images where only local information matters.