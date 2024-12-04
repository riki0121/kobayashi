from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")

results = model("ex2.jpg", save=True, save_txt=True, save_conf=True)
img = cv2.imread("ex2.jpg") #画像読み込み

boxes = results[0].boxes #検出データをboxesに格納

Max_m = 0

for box in boxes: #boxesの中身の一つ一つのboxをループしている（一回目のループの場合、１番上のデータ）

    if int(box.cls) == 0: #人だけを抽出

        x0 = int(box.data[0][0]) #x座標の始点
        y0 = int(box.data[0][1]) #y座標の始点
        x1 = int(box.data[0][2]) #x座標の終点
        y1 = int(box.data[0][3]) #y座標の終点

        m = abs(x1-x0)*abs(y1-y0) #面積を求める

        if Max_m < m: #最大の面積を抽出
            Max_m = m
            Mx0,Mx1,My0,My1 = x0,x1,y0,y1

cv2.rectangle(img,(Mx0,My0),(Mx1,My1),color =(0,0,255),thickness = 3)

print(box.data)

print(results[0].names)

cv2.imshow("表示した画像",img) #画像表示
cv2.waitKey(0) #表示して0秒待機、すなわち無限に
