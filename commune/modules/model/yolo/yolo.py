from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

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
          # label of the boxes.
          "label": r.names[int(r.boxes.cls.numpy()[i])],
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
          "label": r.names[int(r.boxes.cls.numpy()[i])],
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
          # label of indices of the top 5 classes.
          "label": r.names[int(r.probs.top5[i])],
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
  
