import cv2
from image_numpy import *




    # pas
read_img = cv_test()
det = Detection()
## 原圖資訊
read_img.read_img('image004.png')
img_ifo = read_img.img
## HSV圖資訊
read_img.HSV_pictures(read_img.img)
img_HSV_ifo = read_img.img

# read_img.__init__()

def empty(self):
    print(hum_min,hum_max,sat_min,sat_max,val_min,val_max)
det.create_windows('control',600,270)
det.Trackbar('hue min','control',0,179,empty)
det.Trackbar('hue max','control',179,179,empty)
det.Trackbar('sat min','control',0,255,empty)
det.Trackbar('sat max','control',255,255,empty)
det.Trackbar('val min','control',0,255,empty)
det.Trackbar('val max','control',255,255,empty)
print("按下q退出")
while 1:
    hum_min = det.get_value('hue min','control')
    hum_max = det.get_value('hue max','control')
    
    sat_min = det.get_value('sat min','control')
    sat_max = det.get_value('sat max','control')
    
    val_min = det.get_value('val min','control')
    val_max = det.get_value('val max','control')

    low = np.array([hum_min,sat_min,val_min])
    up = np.array([hum_max,sat_max,val_max])
    mask = cv2.inRange(img_HSV_ifo,low,up)
    resvlt = cv2.bitwise_and(img_ifo,img_ifo,mask = mask)
    cv2.waitKey(1)
    cv2.imshow('mask',mask)
    cv2.imshow('resvlt',resvlt)

    if cv2.waitKey(10) == ord('q'):
        print("退出")
        break

# read_img.show_picture()

