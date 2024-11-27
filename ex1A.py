from ultralytics import YOLO
import cv2

model = YOLO("yolov8x-pose.pt")
img = cv2.imread("ex1.jpg") #画像読み込み

results = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data) #.dataの中にx,y座標


points = [(5,6),(5,7),(5,11),(9,7),(6,8),(6,12),(10,8),(13,11),(13,15),(14,12),(14,16),(11,12)]#繋げる（線を引く）  
color = (200,155,255)

for person in keypoints: #personにkeypointsの各要素を格納
    for point_pair in points: #pointsの各要素をpointに格納

        pt1 = (int(person.data[0][point_pair[0]][0]), int(person.data[0][point_pair[0]][1])) 
        pt2 = (int(person.data[0][point_pair[1]][0]), int(person.data[0][point_pair[1]][1]))  #pt1とpt2をつなぐ

        cv2.line(img,pt1,pt2,color,3)


nx = 0
ny = 0

for person in keypoints: #personにkeypointsの各要素を格納
    for joint in range(5,17): #一人なので二重ループの必要なし。

        x = int(person.data[0][joint][0]) #x座標
        y = int(person.data[0][joint][1]) #y座標

        Lsx =int(person.data[0][5][0])
        Lsy =int(person.data[0][5][1])

        Rsx =int(person.data[0][6][0])
        Rsy =int(person.data[0][6][1])
        
        Lhx =int(person.data[0][11][0])
        Lhy =int(person.data[0][11][1])

        Rhx =int(person.data[0][12][0])
        Rhy =int(person.data[0][12][1])


        Newx = (Lsx + Rsx + Lhx + Rhx)/4
        Newy = (Lsy + Rsy + Lhy + Rhy)/4

        nx = int(Newx)
        ny = int(Newy)

        cv2.circle(img,
        center=(x,y),
        radius=3,
        color=(0,0,255),
        thickness=3,
        lineType=cv2.LINE_4,
        shift=0)

cv2.circle(img,center=(nx,ny),radius=3,color=(0,0,255),thickness=3, lineType=cv2.LINE_4,shift=0)

cv2.imshow("表示した画像",img) #画像表示
cv2.waitKey(0) #表示して0秒待機、すなわち無限に
