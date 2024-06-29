import cv2
import numpy as np


img = cv2.imread('image004.png')
img = cv2.resize(img,(0,0),fx=0.25,fy=0.25)ㄆ
Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(Gray,150,200)
contour,b = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
for cnt in contour:
    area = cv2.contourArea(cnt)
    #print(area)    #顯示邊框內的面積
    if area >= 100: #防止噪點
        lenght = cv2.arcLength(cnt,1)
        cv2.drawContours(img,cnt,-1,(255,0,0),4)
        pol = cv2.approxPolyDP(cnt,lenght * 0.02,1)
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
        if len(pol) == 3:
            cv2.putText(img,"triangles",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        elif len(pol) == 4:
            cv2.putText(img,"square",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        elif len(pol) == 5:
            cv2.putText(img,"pentagon",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        elif len(pol) > 5:
            cv2.putText(img,"Circular",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            print("is bug")
    print(len(pol))
cv2.imshow("img",img)
cv2.imshow('canny',canny)
cv2.waitKey(0)