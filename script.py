import matplotlib.pyplot as plt
import math
import numpy as np
import keras_ocr
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path)
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        
        #we can try to change them, for INPAINT_NS we could also use INPAINT_TELEA or INPAINT_CB
        #the kernel size second param can also be adjusted to determine the size of the neighborhood
        img = cv2.inpaint(img, mask, 4, cv2.INPAINT_NS)  
    return(img)

def remove_watermark(img_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    img=inpaint_text(img_path, pipeline)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

if __name__ =="__main__":
    input_folder_path = 'input_folder'
    for filename in os.listdir(input_folder_path):
        img=remove_watermark(f"{input_folder_path}/{filename}")
        cv2.imwrite(f"clean_images/{filename}",img)
    
