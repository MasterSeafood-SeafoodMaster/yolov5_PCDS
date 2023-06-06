import cv2
import time
import torch
import numpy as np
import random
import threading as t
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression


# 要輸出的影片檔案名稱
output_file = 'output.mp4'

# 影片的解析度（寬度和高度）
frame_width = 320
frame_height = 240

# 設定每秒的幀數（FPS）
fps = 30.0
# 使用FourCC編碼器建立VideoWriter物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))


#----------------------------匯入涵式庫----------------------------

device = select_device("")
imgsz = (640, 640)
conf_thres = 0.662
iou_thres = 0.5
classes = None
agnostic_nms = 2
max_det = 10
ud_y = [175, 305]

def IoU(box1, box2):
    Area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    Area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    if ( min(box1[2], box2[2]) - max(box1[0], box2[0]) < 0) or (min(box1[3], box2[3]) - max(box1[1], box2[1])  <0 ):
        return 0 
    Intersection =(min(box1[2], box2[2]) - max(box1[0], box2[0]))*(min(box1[3], box2[3]) - max(box1[1], box2[1]))
    #Union = Area1 + Area2 - Intersection
    Union = Area1
    return Intersection/Union #計算Intersection over Union

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
		pred[i].append((0, 0, 0))
		pred[i].append([])
	return pred #將yolo predict轉為list且回傳

def getCenter(box):
	return [(box[0]+box[2])//2, (box[1]+box[3])//2] #回傳中心點
#----------------------------定義涵式----------------------------

model = DetectMultiBackend("./custom_model/best.pt", device=device, dnn=False, data="./data/baseball_dataset.yaml", fp16=False)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz)) #讀取Model

cap = cv2.VideoCapture("./videos/F2.avi") #讀取影片
vLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #計算Frame總數

lastPred=[]; index=1; ud_mat=[0, 0]
for f in range(vLength):
	start = time.time()
	ret, frame = cap.read()
	frame = cv2.resize(frame, (640, 480))
	pred = yoloPred(frame, model)
	for bbox in pred:
		lf=[]
		for lbbox in lastPred:
			if IoU(bbox, lbbox) > 0.5:
				bbox[4] = lbbox[4]
				bbox[6] = lbbox[6]
				lbbox[7].append(getCenter(bbox)) #儲存歷史路徑
				bbox[7] = lbbox[7]
				lf = lbbox
				lastPred.remove(lbbox)
				break
		if bbox[4]==0:
			bbox[4] = index
			index += 1
		if bbox[6]==(0, 0, 0):
			bbox[6]=(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)) #加入隨機顏色

		if len(lf)>0:
			if getCenter(lf)[1]>ud_y[0] and getCenter(bbox)[1]<=ud_y[0]:
				ud_mat[0] += 1
			elif getCenter(lf)[1]<ud_y[1] and getCenter(bbox)[1]>=ud_y[1]:
				ud_mat[1] += 1

	for rebbox in lastPred:
		rebbox[5]+=1
		if rebbox[5]<5: #當遺失累計>=5時，移除bounding box
			pred.append(rebbox)
	#----------------------------修正Predict list為 [左上x, 左上y, 右下x, 右下y, box編號, 遺失累計, 顏色, 軌跡]----------------------------

	for bbox in pred:
		#frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
		#frame = cv2.putText(frame, str(bbox[4]), getCenter(bbox), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
		for path in range(len(bbox[7])-1):
			frame = cv2.line(frame, bbox[7][path], bbox[7][path+1], bbox[6], 2)

		frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[6], 2)
		frame = cv2.putText(frame, str(bbox[4]), getCenter(bbox), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox[6], 1, cv2.LINE_AA)

	frame = cv2.line(frame, (0, ud_y[0]), (640, ud_y[0]), (0, 0, 255), 2)
	frame = cv2.line(frame, (0, ud_y[1]), (640, ud_y[1]), (0, 0, 255), 2)
	frame = cv2.putText(frame, "get off: "+str(ud_mat[0]), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	frame = cv2.putText(frame, "get on: "+str(ud_mat[1]), (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	#----------------------------繪圖----------------------------
	
	lastPred = pred
	o = cv2.resize(frame, (320, 240))
	out.write(o)
	cv2.imshow("live", frame)
	end = time.time()
	print(str(end-start), "s")
	
	cv2.waitKey(1)
	if cv2.waitKey(1) == ord('q'):
		break
out.release()
cv2.destroyAllWindows()

#----------------------------顯示----------------------------