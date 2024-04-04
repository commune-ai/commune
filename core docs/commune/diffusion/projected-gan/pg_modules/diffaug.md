# Differential Augmentation for Data-Efficient GAN Training

This Python module implements differential augmentations for data-efficient GAN training, as proposed in a paper by Shengyu Zhao et al. This method uses various augmentations such as brightness, saturation, contrast, translation, and cutout on the input data to make the GAN model more robust and perform well even with less data.

Key Features
=============

- `DiffAugment(x, policy='', channels_first=True)`: Function to apply differential augmentation on data according to the given policy. It processes the data one augmentation policy at a time. The `policy` input is a string containing one or more policies separated by comma.

- Differential augmentation functions: Each function takes input data and performs a specific augmentation. The functions include:

  - `rand_brightness(x)`: Randomly changes brightness of the data
  - `rand_saturation(x)`: Randomly changes saturation of the data
  - `rand_contrast(x)`: Randomly changes contrast of the data
  - `rand_translation(x, ratio=0.125)`: Randomly translates the data
  - `rand_cutout(x, ratio=0.2)`: Randomly applies cutout augmentation

- `AUGMENT_FNS`: This dictionary maps the augmentation policy names to their corresponding function(s). It is used to look up the functions for a given policy.

Example Usage
=============

```python
import torch

# Load your data here
data = torch.randn(10, 3, 256, 256)

# Apply DiffAugment with 'color,translation' policy
augmented_data = DiffAugment(data, policy='color,translation')
```

Above example applies color and translation differential augmentation to the data.

Please note, all the augmentation functions work with data in `NCHW` format (batch size, number of channels, height, width).

Requirements
============

This module requires PyTorch library.