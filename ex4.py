import cv2
from ultralytics import YOLO
import numpy as np
import math

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)
count = 0

#画像ファイル
img = cv2.imread("ex1.jpg") #画像読み込み
results_img = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints_img = results_img[0].keypoints
print(keypoints_img.data) #.dataの中にx,y座標


dis = 0 #初期化
diff_list = [] #差を格納するリスト


# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        dis = 0

        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model(frame, save=True)
        keypoints = results[0].keypoints

        color = (255,0,0)

        
        for i in range(0,17): #pointsの各要素をpointに格納

            x = int(keypoints.data[0][i][0])
            y = int(keypoints.data[0][i][1]) #pt1とpt2をつなぐ

            x_img = int(keypoints_img.data[0][i][0]) 
            y_img = int(keypoints_img.data[0][i][1])   #画像の座標を求める

            dis += math.sqrt((x - x_img)**2 + (y - y_img)**2)


        diff_list.append(dis) #差をリストに格納


        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8トラッキング", frame)

        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break

    count += 1

min_diff = min(diff_list)
min_diff_index = diff_list.index(min_diff)
print("一致したフレーム:",min_diff_index)




# ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
