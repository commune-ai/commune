from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import commune as c

class Yolo(c.Module):

  def __init__(self, cache_key:bool = True):
    self.detectModel = YOLO("yolov8n.pt")
    self.segmentModel = YOLO("yolov8n-seg.pt")
    self.classifyModel = YOLO("yolov8n-cls.pt")
    self.poseModel = YOLO("yolov8n-pose.pt")

  def set_api_key(self, api_key:str, cache:bool = True):
    if api_key == None:
        api_key = self.get_api_key()

    self.api_key = api_key
    if cache:
        self.add_api_key(api_key)

    assert isinstance(api_key, str)

  # identifying the location and class of objects in an image
  def detect(self, image: str = "test_image.png"):
    results = self.detectModel(image)

    response = []
    for r in results:
      for i in range(len(r.boxes.cls.numpy())):
        response.append({
          # label   of the boxes.
          "label " : r.names[int(r.boxes.cls.numpy()[i])],
          # confidence values of the boxes.
          "confidence": r.boxes.conf.numpy()[i],
          # boxes in xywh format.
          "xywh": r.boxes.xywh.numpy()[i],
        })

    return response
  
  # identifying individual objects in an image and segmenting them from the rest of the image.
  def segment(self, image: str = "test_image.png"):
    results = self.segmentModel(image)

    response = []
    for r in results:
      for i in range(len(r.boxes.cls.numpy())):
        response.append({
          "label " : r.names[int(r.boxes.cls.numpy()[i])],
          "confidence": r.boxes.conf.numpy()[i],
          "xywh": r.boxes.xywh.numpy()[i],
          # A list of segments in pixel coordinates represented as tensors.
          "xy_sz": len(r.masks.xy[i]),
        })

    return response
  
  # classifying an entire image into one of a set of predefined classes.
  def classify(self, image: str = "test_image.png"):
    results = self.classifyModel(image)

    response = []
    for r in results:
      for i in range(len(r.probs.top5)):
        response.append({
          # label   of indices of the top 5 classes.
          "label " : r.names[int(r.probs.top5[i])],
          # Confidences of the top 5 classes.
          "confidence": r.probs.top5conf.numpy()[i],
        })

    return response
  
  # identifying the location of specific points in an image, usually referred to as keypoints.
  def pose(self, image: str = "test_image.png"):
    results = self.poseModel(image)

    response = []
    for r in results:
      for i in range(len(r.keypoints.xy.numpy())):
        response.append({
          # A list of keypoints in pixel coordinates represented as tensors.
          "keypoint": r.keypoints.xy.numpy()[i],
          # confidence values of keypoints if available, else None.
          "confidence": r.keypoints.conf.numpy()[i],
        })

    return response
  
  def gradioDetect(self, image = "test_image.png"):
    results = self.detectModel(image)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        return Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    
  def gradioSegment(self, image = "test_image.png"):
    results = self.segmentModel(image)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        return Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    
  def gradioClassify(self, image = "test_image.png"):
    results = self.classifyModel(image)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        return Image.fromarray(im_array[..., ::-1])  # RGB PIL image
  def gradioPose(self, image = "test_image.png"):
    results = self.poseModel(image)

    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        return Image.fromarray(im_array[..., ::-1])  # RGB PIL image

  def gradio(self):
    with gr.Blocks() as demo:
      with gr.Row():
        # input; image
        with gr.Column():             
          img_in = gr.Image(label = "image", type = "filepath")
        # output; 4 tabs each contain a button and a image output
        with gr.Column():
          with gr.Tab("Detection"):
            detect_but = gr.Button("Detect")
            detect_out = gr.Image(label = "result")   
          with gr.Tab("Segmentation"):
            segment_but = gr.Button("Segment")
            segment_out = gr.Image(label = "result")   
          with gr.Tab("Classification"):
            classify_but = gr.Button("Classify")
            classify_out = gr.Image(label = "result")   
          with gr.Tab("Pose"):
            pose_but = gr.Button("Pose")
            pose_out = gr.Image(label = "result")     
        # Bind function to each button
        detect_but.click(fn = self.gradioDetect, inputs = img_in, outputs = detect_out)
        segment_but.click(fn = self.gradioSegment, inputs = img_in, outputs = segment_out)
        classify_but.click(fn = self.gradioClassify, inputs = img_in, outputs = classify_out)
        pose_but.click(fn = self.gradioPose, inputs = img_in, outputs = pose_out)

    demo.launch(quiet=True, share=True)
