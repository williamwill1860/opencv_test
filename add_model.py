import cv2

from image_numpy import *
from camara import *


camara_function = camara()

cam = camara_function.open_camara(0)                           #開啟鏡頭
eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')

frame = 2#方框的寬度
print("按下Q退出鏡頭")
while 1:
    chuck_video,img = cam.read()
    
    if cv2.waitKey(10) == ord('q'):
        print("exit")
        break 
    if chuck_video:
        img = cv2.resize(img,(500,500))
        Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        eye_return = eye_cascade.detectMultiScale(Gray,1.1,30)
        eye_num = 0 
        for (x,y,w,h) in eye_return:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),frame)#frame代表框框的寬度
            eye_num = eye_num+1
            
            print("x1=",x,"y1=",y,"w1=",w,"h1=",h)
            cut_img = img[y+frame:y+h-frame,x+frame:x+w-frame]
            cut_img = cv2.resize(cut_img,(500,500))
            
            # Gray = cv2.cvtColor(cut_img,cv2.COLOR_BGR2GRAY)
            # Gray = cv2.GaussianBlur(Gray,(3,3),0)
            # canny = cv2.Canny(Gray,50,60)
            # # cut_img = cv2.line(cut_img,(0,255),(500,255),(0,0,255),2)
            # # cut_img = cv2.line(cut_img,(255,0),(255,500),(0,0,255),2)
            cv2.imshow("cut_img",cut_img)
            # cv2.imshow("Gray",Gray)
            # cv2.imshow("canny",canny)
            # if contour:
            #     cv2.imshow("contour",contour)
            cv2.imshow(f"{eye_num}cut_img",cut_img)
        if cv2.waitKey(10) == ord('q'):
            print("exit")
            break
        cv2.imshow('Press q to exit the program',img)
        cv2.waitKey(1)
        # print("x1=",x1)
        
    else:
        print("no camara")
        break



