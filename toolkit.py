import numpy as np
import torch
import cv2

from utils.augmentations import letterbox
from utils.general import non_max_suppression
imgsz = (640, 640)
conf_thres = 0.662
iou_thres = 0.5
classes = None
agnostic_nms = 2
max_det = 10

def IoU(box1, box2):
    Area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    Area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    if ( min(box1[2], box2[2]) - max(box1[0], box2[0]) < 0) or (min(box1[3], box2[3]) - max(box1[1], box2[1])  <0 ):
        return 0 
    Intersection =(min(box1[2], box2[2]) - max(box1[0], box2[0]))*(min(box1[3], box2[3]) - max(box1[1], box2[1]))
    #Union = Area1 + Area2 - Intersection
    Union = Area1
    return Intersection/Union

def yoloPred(frame, yolo_model):
	im = frame.copy()
	im = letterbox(im, imgsz, 32, True)[0]
	im = im.transpose((2, 0, 1))[::-1]
	im = np.ascontiguousarray(im)
	im = torch.from_numpy(im).to(device)
	im = im.half() if yolo_model.fp16 else im.float()
	im /= 255
	if len(im.shape) == 3:im = im[None]
	pred = yolo_model(im, augment=False, visualize=False)
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

	pred = pred[0].tolist()
	for i in range(len(pred)):
		for j in range(len(pred[i])):
			pred[i][j] = int(pred[i][j])

	return pred