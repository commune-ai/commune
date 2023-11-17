from ultralytics import YOLO
import numpy as np
import cv2

import commune as c

class Yolo(c.Module):

  def __init__(self, cache_key:bool = True):
    self.model = YOLO("yolov8n.pt")

  def set_api_key(self, api_key:str, cache:bool = True):
      if api_key == None:
          api_key = self.get_api_key()

      self.api_key = api_key
      if cache:
          self.add_api_key(api_key)

      assert isinstance(api_key, str)

  def predict(self,
    model:str = "yolov8n.pt", # model
    source: str = "test_image.png"   # test image file path
    ):
    if model != "yolov8n.pt":
      self.model = YOLO(model)

    output_layers = self.model.predict(source=source)
    detection_info = []
    names = output_layers[0].names

    # Loop over the detections
    for detection in output_layers[0]:
      x, y, w, h = detection.boxes.xywh[0].cpu().numpy().squeeze()
      class_id = detection.boxes.cls.cpu().numpy().squeeze().item()
      confidence = detection.boxes.conf.cpu().numpy().squeeze().item()
      detection_info.append({
          'x': x,
          'y': y,
          'w': w,
          'h': h,
          'class': int(class_id),
          'label': names[int(class_id)],
          'confidence': float(confidence)
      })

    return detection_info