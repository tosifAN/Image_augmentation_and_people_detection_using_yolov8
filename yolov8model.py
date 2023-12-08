
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# 0 for person class

src='source/4.mp4'

results = model(src,save=True, project="runs/detect", name="inference", exist_ok=True,classes=0)  # predict on an image
