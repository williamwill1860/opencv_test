import cv2
import numpy as np

from image_numpy import *
from camara import *

cv_function = cv_test()
camara_function = camara()
# camara_function.open_camara()                           #開啟鏡頭
cv_function.read_img('image001.jpg')
# cv_function.draw_picture(500,500,0,0,0)
# img = cv_function.draw_line(cv_function.img,100,100,300,300,255,0,0,3)

# cv_function.draw_rect(cv_function.img,100,100,300,300,255,0,0,2)

# cv_function.draw_circle(cv_function.img,100,100,30,255,0,0,2)
cv_function.draw_word(cv_function.img,'Handsome boy ',100,400,cv2.FONT_HERSHEY_COMPLEX,1,200,125,125,2)

cv_function.Edge_Detection(cv_function.img,150,200)    #圖片資訊,Edgemin,Edgemax，邊緣
cv_function.HSV_pictures(cv_function.img)

# cv_function.dilate_function(cv_function.canny,3,1)      #圖片資訊,kernel大小,膨脹次數

# cv_function.erode_function(cv_function.dilate,5,1)   #圖片資訊,kernel大小,收縮次數
cv_function.show_picture()
# cv_function.Edge_Detection(cv_function.img,15,0)    #圖片資訊,kernel,標準差
# print(type(img))
