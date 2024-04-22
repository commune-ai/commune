# Convert Image to Video Module - README


This module is designed for converting image to video. It offers a `gradio` function that users can convert the image.
`gradio` function offers user-interface to users so that users can enjoy this module. Also I added docstring so that you can easily commit.

#### install dependencies

```
pip install -r requirements.txt
```

#### Input:
User upload several images using upload element

#### Output:
The video will out using that images

### Usage Example:

```c model.image2video gradio```

### Fixing bugs
In line 50 of model_image2video.py, replace Image.ANTIALIAS with Image.LANCZOS.
In site-package/image_tools/sizes.py, replace ANTIALIAS with LANCZOS.
