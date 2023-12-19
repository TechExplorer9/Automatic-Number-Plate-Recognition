Author @Atharva Vitekar

# Automatic Number Plate Recognition (ANPR) with EasyOCR and YOLOv3

This Python script performs Automatic Number Plate Recognition (ANPR) using EasyOCR for text recognition and YOLOv3 for object detection. It detects number plates in either a live video stream or an input image.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3.x
- OpenCV
- EasyOCR
- NumPy

You can install the required libraries using the following command:

```bash
pip install easyocr opencv-python numpy

Additionally, you need the YOLOv3 weights and configuration file. Download them from the official YOLO website:

Download Link given below
YOLOv3 Weights (yolov3.weights):  https://pjreddie.com/darknet/yolo/ 
YOLOv3 Configuration File (yolov3.cfg): https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
Place the weights and configuration files in the same directory as the script.