import cv2

B0c = cv2.VideoCapture("./B0.mp4")
B1c = cv2.VideoCapture("./B1.mp4")
F0c = cv2.VideoCapture("./F0.mp4")
F1c = cv2.VideoCapture("./F1.mp4")

lList = [int(B0c.get(cv2.CAP_PROP_FRAME_COUNT)), int(B1c.get(cv2.CAP_PROP_FRAME_COUNT)), int(F0c.get(cv2.CAP_PROP_FRAME_COUNT)), int(F1c.get(cv2.CAP_PROP_FRAME_COUNT))]

# 要輸出的影片檔案名稱
output_file = 'comb.mp4'

# 影片的解析度（寬度和高度）
frame_width = 640
frame_height = 480

# 設定每秒的幀數（FPS）
fps = 30.0

# 使用FourCC編碼器建立VideoWriter物件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
oout = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

for i in range(min(lList)):
	ret, B0f = B0c.read()
	ret, B1f = B1c.read()
	ret, F0f = F0c.read()
	ret, F1f = F1c.read()

	c0 = cv2.hconcat((B0f, F0f))
	c1 = cv2.hconcat((F1f, B1f))

	com = cv2.vconcat((c0, c1))
	oout.write(com)

	cv2.imshow("l", com)
	cv2.waitKey(1)

oout.release()
