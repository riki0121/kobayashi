import cv2
from ultralytics import YOLO
import numpy as np
import math

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)

# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model(frame, save=True)
        keypoints = results[0].keypoints

        points = [(5,6),(5,7),(5,11),(9,7),(6,8),(6,12),(10,8),(13,11),(13,15),(14,12),(14,16),(11,12)]#繋げる（線を引く
        R_points = [(6,8),(8,10)] #右腕のみのポイント
        color = (255,0,0)

        R_shoulder =(keypoints.data[0][6][0] ,keypoints.data[0][6][1]) 
        R_elbow = (keypoints.data[0][8][0] ,keypoints.data[0][8][1]) 
        R_hip = (keypoints.data[0][12][0] ,keypoints.data[0][12][1]) 
        R_wrist =  (keypoints.data[0][10][0] ,keypoints.data[0][10][1]) 

        vec_a = (R_elbow[0]-R_shoulder[0], R_elbow[1]-R_shoulder[1])
        vec_c = (R_hip[0]-R_shoulder[0], R_hip[1]-R_shoulder[1])


        A_vec = math.sqrt(vec_a[0]**2 + vec_a[1]**2)
        C_vec = math.sqrt(vec_c[0]**2 + vec_c[1]**2)

        inner = vec_a[0] * vec_c[0] + vec_a[1] *  vec_c[1]

        cos = inner / (A_vec * C_vec)

        angle = math.degrees(math.acos(cos))

        
        for point_pair in points: #pointsの各要素をpointに格納

            pt1 = (int(keypoints.data[0][point_pair[0]][0]), int(keypoints.data[0][point_pair[0]][1])) 
            pt2 = (int(keypoints.data[0][point_pair[1]][0]), int(keypoints.data[0][point_pair[1]][1]))  #pt1とpt2をつなぐ

            cv2.line(frame,pt1,pt2,color,3)

        if 80 <= angle <= 100:

            for point_pair in R_points:
                pt3 = (int(keypoints.data[0][point_pair[0]][0]), int(keypoints.data[0][point_pair[0]][1])) 
                pt4 = (int(keypoints.data[0][point_pair[1]][0]), int(keypoints.data[0][point_pair[1]][1])) 
                
                cv2.line(frame,pt3,pt4,(0,0,255),3)



        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8トラッキング", frame)

        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break

# ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
