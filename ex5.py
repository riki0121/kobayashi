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

         # 対象領域を切り出す
        roi = img[y0:y1, x0:x1]
        # 青色を抽出するためにHSV空間に変換
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        #青色の範囲を指定 (HSVでの青色範囲)
        lower_blue = (110, 90, 50)  # 青色の下限値
        upper_blue = (140, 255, 255)  # 青色の上限値        

        # 青色をマスクで抽出
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        print(cv2.countNonZero(mask))
        # 青色の割合が一定以上なら赤線で囲む 
        if cv2.countNonZero(mask) > 1: # しきい値を適切に調整 
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

print(box.data)

print(results[0].names)

cv2.imshow("表示した画像",img) #画像表示
cv2.waitKey(0) #表示して0秒待機、すなわち無限に
