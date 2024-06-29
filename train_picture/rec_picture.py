import cv2
import numpy as np
import os

def rename(path,r_path):
    os.chdir(path)
    i = 300
    for filename in os.listdir(path):
        portion = os.path.splitext(filename)
        j=str(i)
        new_name = j + portion[1]
        os.rename(filename,new_name)
        i = i + 1
    i = 1
    for filename in os.listdir(path):
        portion = os.path.splitext(filename)
        j=str(i)
        new_name = j + portion[1]
        os.rename(filename,new_name)
        i = i + 1
    os.chdir(r_path)
def pick_picture(path,picture_mun):     #選取圖片
    
    try:
        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))
        return img
    except:
        print("total read picture = ",picture_mun-1)
        return 0
def draaw_rect(img,eye_cascade):    #畫上紅色框框
    frame = 2
    num=0
    Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eye_return = eye_cascade.detectMultiScale(Gray,1.1,30)
    for (x,y,w,h) in eye_return:
        num = num+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),frame)#frame代表框框的寬度
    return img,num
def revolve_picture(img,r):   #旋轉180度
    row = col = 224
    r = cv2.getRotationMatrix2D((row/2,col/2),r,1)
    revole = cv2.warpAffine(img,r,(row,col))
    return revole
def filp_hor(img):     # 水平翻轉
    filp_img = cv2.flip(img,1)
    return filp_img
def blurs_picture(img,kernal_size): #模糊
    blurs = cv2.GaussianBlur(img,(kernal_size,kernal_size),0)
    return blurs

def main(dir_name):
    
    picture_mun = 0
    path = f"D:/Information_Institute_program/opencv_test/cheat/{dir_name}"
    r_path = "D:/Information_Institute_program/opencv_test/cheat"
    # os.chdir(r_path)
    rename(path,r_path)
    for i in range(1,100):
        picture_mun = picture_mun +1
        img = pick_picture(f"{dir_name}/{i}.jpg",picture_mun)   #選出圖片
        if(type(img)==int):                                     #如果回傳是0(int) 代表已經沒有圖片了
            print(f"{dir_name} done ...")
            break
        rect_img,eye_num = draaw_rect(img,eye_cascade)          #畫方框
        re_img = revolve_picture(img,r=5)                           #旋轉圖片(180度)
        r_img = revolve_picture(img,r=10)                           #旋轉圖片(180度)
        # filp_img = filp_hor(img)                                #鏡像圖片
        blurs = blurs_picture(img,kernal_size = 9)
        cv2.imwrite(f"../cheat/{dir_name}/{i}.jpg",rect_img)             #圖片存檔
        # cv2.imwrite(f"../cheat/{dir_name}/{i}filp_img.jpg",filp_img)     #圖片存檔
        cv2.imwrite(f"../cheat/{dir_name}/blurs{i}.jpg",blurs)     #圖片存檔
        cv2.imwrite(f"../cheat/{dir_name}/revolve{i}.jpg",re_img)
        cv2.imwrite(f"../cheat/{dir_name}/revolve_90_{i}.jpg",r_img)

        if eye_num == 0:
            # print(f"{i}.jpg is no Detection eye")             #如果找不到眼睛的區塊就進入這裡
            pass
        else:
            cv2.imwrite(f"{dir_name}/Detection{i}.jpg",rect_img)    #找得到眼睛的圖片就將劃了框框的圖片存檔
        # cv2.imshow(f'{i}img',rect_img)
        cv2.waitKey(1)  
        
dirname = ["down","left","up","right"]  #要進入的資料夾名稱
eye_cascade = cv2.CascadeClassifier('../opencv/data/haarcascades/haarcascade_eye.xml')  #load模型
for name in dirname:
    main(name)