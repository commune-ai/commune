# Yolo v8 Module - README

Introducing Ultralytics [YOLOv8](https://docs.ultralytics.com/), the latest version of the acclaimed real-time object detection and image segmentation model. YOLOv8 is built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

## Using the Module

### Detect

`c model.yolo detect image='image.png'`

Identifying the location and class of objects in an image

### Segment

`c model.yolo segment image='image.png'`

Identifying individual objects in an image and segmenting them from the rest of the image.

### Classify

`c model.yolo classify image='image.png'`

Classifying an entire image into one of a set of predefined classes.

### Pose

`c model.yolo pose image='image.png'`

Identifying the location of specific points in an image, usually referred to as keypoints.

## Gradio UI

`c model.yolo gradio`

Testing the module on Gradio UI.