import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import tensorflow as tf

import numpy as np
import argparse
from image_numpy import *
from camara import *
from tensorflow.lite.python.interpreter import Interpreter
'''
    install:pip install tensorflow
            pip install tflite-runtime
    windows run cammon : python read_video.py --model model/model.tflite --labels model/labels.txt  
    rspi run cammon : python3 read_video.py --model model/model.tflite --labels model/labels.txt  
'''

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}
    
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    # ordered = np.max(output)
    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    input_num = "7"         #輸入第幾支影片
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='model/model.tflite', required=True)
    parser.add_argument('--labels', help='model/labels.txt', required=True)
    args = parser.parse_args()
    labels = load_labels(args.labels)
    camara_function = camara()
    cam = camara_function.open_camara(f"input{input_num}.mp4")                           #開啟鏡頭
    eye_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_eye.xml')
    # eye_cascade = cv2.CascadeClassifier('model/eye.xml')
    
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    _,height, width,_ = interpreter.get_input_details()[0]['shape']

    counter = 0
    frame = 2#方框的寬度
    print("按下Q退出鏡頭")
    while 1:
        chuck_video,img = cam.read()
        if cv2.waitKey(10) == ord('q'):
            print("exit")
            break 
        if chuck_video:
            eye_return = eye_cascade.detectMultiScale(img,1.1,30)
            # Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            for (x,y,w,h) in eye_return:
                
                counter = counter + 1
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),frame)#frame代表框框的寬度 畫框框
                cut_img = img[y+frame:y+h-frame,x+frame:x+w-frame]  #裁切圖片
                cut_img = cv2.resize(cut_img,(height,width))            #將圖片改為預先設定好的大小(224*224)
                results = classify_image(interpreter, cut_img)      #分類
                label_id, prob  = results[0]
                # Gray = cv2.cvtColor(cut_img,cv2.COLOR_BGR2GRAY)
                # Gray = cv2.GaussianBlur(Gray,(3,5),2)
                # canny = cv2.Canny(Gray,20,50)
                print(labels[label_id],prob)
                # cv2.imwrite(f"cheat/ans/{labels[label_id]}/{counter}_probs.jpg",cut_img) #寫答案
                # cv2.imwrite(f"cheat/{counter}_{input_num}.jpg",cut_img)                   #給人工分類
                cv2.putText(cut_img,labels[label_id] + " " + str(round(prob,3)), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                
                cv2.imshow("cut_img",cut_img)
            cv2.imshow("video",img)
            # cv2.imshow("Gray",Gray)
            # cv2.imshow("canny",canny)
            # cv2.waitKey(1)
            # print("total picture = ",counter)
        else:
            print("no video")
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()



