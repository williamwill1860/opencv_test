import cv2
import numpy as np

img1 = cv2.imread("1.png")
img2 = cv2.imread("2.png")
img3 = cv2.imread("3.png")
img4 = cv2.imread("4.png")
img1 = cv2.resize(img1,(183,183))
img2 = cv2.resize(img2,(183,183))
img3 = cv2.resize(img3,(183,183))
img4 = cv2.resize(img4,(183,183))
img_final = cv2.vconcat([img1,img2,img3,img4])
cv2.imshow("img",img_final)
cv2.imwrite("final_2.png",img_final)
cv2.waitKey(0)