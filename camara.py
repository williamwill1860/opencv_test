import cv2

class camara():
    def __init__(self) -> None:
        pass
    def open_camara(self,cam_num):
        cam = cv2.VideoCapture(cam_num)
        return cam
        

            
            