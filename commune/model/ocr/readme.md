# OCR Module - README

## Introduction

We use EasyOCR in this module.
[EasyOCR](https://www.jaided.ai/easyocr/) is a python module for extracting text from image. It is a general OCR that can read both natural scene text and dense text in document. We are currently supporting 80+ languages and expanding.

## How to Use OCR Module?

### readtext

`c model.ocr readtext image='example.png'`

Method for reading text from the image.

*Parameters*

    - image (string, numpy array, byte) - Input image
    - decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
    - beamWidth (int, default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
    - batch_size (int, default = 1) - batch_size>1 will make EasyOCR faster but use more memory
    - workers (int, default = 0) - Number thread used in of dataloader
    - allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
    - blocklist (string) - Block subset of character. This argument will be ignored if allowlist is given.
    - detail (int, default = 1) - Set this to 0 for simple output
    - paragraph (bool, default = False) - Combine result into paragraph
    - min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
    - rotation_info (list, default = None) - Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.

*Return*

    list of results

### detect

`c model.ocr detect image='example.png'`

Method for detecting text boxes.

*Parameters*

    - image (string, numpy array, byte) - Input image
    - min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
    - text_threshold (float, default = 0.7) - Text confidence threshold
    - low_text (float, default = 0.4) - Text low-bound score
    - link_threshold (float, default = 0.4) - Link confidence threshold
    - canvas_size (int, default = 2560) - Maximum image size. Image bigger than this value will be resized down.
    - mag_ratio (float, default = 1) - Image magnification ratio
    - slope_ths (float, default = 0.1) - Maximum slope (delta y/delta x) to considered merging. Low value means tiled boxes will not be merged.
    - ycenter_ths (float, default = 0.5) - Maximum shift in y direction. Boxes with different level should not be merged.
    - height_ths (float, default = 0.5) - Maximum different in box height. Boxes with very different text size should not be merged.
    - width_ths (float, default = 0.5) - Maximum horizontal distance to merge boxes.
    - add_margin (float, default = 0.1) - Extend bounding boxes in all direction by certain value. This is important for language with complex script (E.g. Thai).
    - optimal_num_chars (int, default = None) - If specified, bounding boxes with estimated number of characters near this value are returned first.

*Return*

    horizontal_list, free_list - horizontal_list is a list of regtangular text boxes. The format is [x_min, x_max, y_min, y_max]. free_list is a list of free-form text boxes. The format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].

## Gradio UI

`c model.ocr gradio`

Testing the module on Gradio UI.