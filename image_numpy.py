import cv2
import numpy as np

class cv_test():
    def __init__(self):
        print("init...")
        self.img = None
        self.canny = None
        self.dilate = None
        cv2.destroyAllWindows()
    def read_img(self,img):
        self.img = cv2.imread(img)
        self.img= cv2.resize(self.img,(500,500))
        cv2.imshow("Original Picture",self.img)
    def cut_img(self,img):
        #=========切割圖片=========================
        cutimg = img[0:200,0:200]

        cv2.imshow("Original Picture",img)
        cv2.imshow("cut picture",cutimg)

        
    
#========畫一張單格的色塊==================
    def draw_picture(self,x,y,b,g,r):
        self.img = np.empty((x,y,3)) #長寬300，BGR3個顏色

        for row in range(x):
            for col in range(y):
                self.img[row][col] = [b,g,r]

        cv2.imshow("this is red picture",self.img)

#=======邊緣檢測==============
    def Edge_Detection(self,img,Edgemin,Edgemax):

        self.canny = cv2.Canny(img,Edgemin,Edgemax)
        cv2.imshow('img',img)
        cv2.imshow(f"canny : (min{Edgemin},max{Edgemax})",self.canny)
#========灰階、模糊=============
    def gray_and_blur(self,img,kernel,offset):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img,(kernel,kernel),offset) #模糊
        cv2.imshow("img",img)
        cv2.imshow("gray",gray)
        cv2.imshow("blur",blur)

#========膨脹=============
    def dilate_function(self,img,kernel_size,times):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        self.dilate = cv2.dilate(img,kernel,times)
        cv2.imshow(f"dilate kernel_size:{kernel_size}times:{times}",self.dilate)
    def show_picture(self):
        print("按下q退出")
        while 1:
            if cv2.waitKey(10) == ord('q'):
                print("退出")
                break

#=======在圖片上畫線==========
    def draw_line(self,img,x1,x2,y1,y2,b,g,r,px):
        line_pictures  = cv2.line(img,(x1,x2),(y1,y2),(b,g,r),px)
        cv2.imshow("draw line",line_pictures )

#=======在圖片上畫方========
    def draw_rect(self,img,x1,x2,y1,y2,b,g,r,px):
        rect_pictures  =cv2.rectangle(img,(x1,x2),(y1,y2),(b,g,r),px)
        cv2.imshow("draw rect",rect_pictures )

#=======在圖片上畫圓========
    def draw_circle(self,img,x1,x2,r,b,g,red,px):
        circle_pictures  =cv2.circle(img,(x1,x2),r,(b,g,red),px)
        cv2.imshow("draw rect",circle_pictures )

#=======在圖片上寫字========
    def draw_word(self,img,word,x,y,font,size,b,g,r,px):
        word_pictures  = cv2.putText(img,word,(x,y),font,size,(b,g,r),px)
        
        cv2.imshow("word pictures ",word_pictures )

#========收縮=============
    def erode_function(self,img,kernel_size,times):
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        erode = cv2.erode(img,kernel,times)
        cv2.imshow(f"dilate kernel_size:{kernel_size}times:{times}",erode)

    def HSV_pictures (self,img):
        HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV pictures ",HSV)

class Detection():
    def __init__(self) -> None:
        pass
    def create_windows(self,window_name,width,hight):
        cv2.namedWindow(window_name)
        cv2.resizeWindow(window_name,width,hight)
    def Trackbar(self,window_name,control_window,init,M,function_name):        
        cv2.createTrackbar(window_name,control_window,init,M,function_name)
            
    def get_value(self,control_line,window_name):
        val = cv2.getTrackbarPos(control_line,window_name)
        return val