# Fourier Spectrum Processing for 2D data

This Python script includes a collection of functions to perform Fourier transformations and other mathematical operations on 2D data. The script is specifically designed to handle batch of 2D images. It uses the popular open-source library `torch` from PyTorch to conduct these operations.

The key features of this script include:

- Computing the Fourier transform with normalization of batch of 2D images with `batch_fft` function.
- Shifting the low frequencies to center of the Fourier transform using `roll_quadrants` function. Frequency shifting can be also performed backwards.
- Computing the azimuthal (circular or radial) averages around the center of an image using `azimuthal_average` function.
- Getting the power spectrum of the image using `get_spectrum` function. It involves Fourier transform, power spectrum calculation, frequency roll and azimuthal averaging.
- Plotting the mean and standard deviation of the calculated spectra using `plot_std` function.

These functions can be used to perform spectral analysis of images data. They can be particularly useful in scientific fields such as Astrophysics, Geosciences and Computational Fluid Dynamics where frequency-space analysis of spatial data is often required.

Please note:
- The script assumes that input data is 2D.
- Functions in this script use PyTorch functions and return PyTorch tensors.
- All functions demand the data to be in a certain format, and throw AttributeError if format is incorrect.

To utilize these functions, import necessary ones into your Python environment and call them on your data. For example:

``` python
import torch
import some_other_libraries

from this_script import azimuthal_average, batch_fft, roll_quadrants

# your data loading and pre-processing
data = torch.rand(100, 128, 128)

# Fourier transform
freq_data = batch_fft(data)

# Shifting frequencies
freq_data_shifted = roll_quadrants(freq_data)

# Azimuthal averaging
avg_freq_data = azimuthal_average(freq_data_shifted)

# Further processing and analysis
# ...
```
It's important to ensure all dependencies (such as `matplotlib` for `plot_std` function) are installed before running the script.