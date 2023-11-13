# Source: https://github.com/xdshang/real-time-object-detection/blob/master/utils.py

import cv2

def draw_result(img, result, scale=1.0):
    img_cp = img.copy()
    for i in range(len(result)):
        x = int(result[i][1]*scale)
        y = int(result[i][2]*scale)
        w = int(result[i][3]*scale)//2
        h = int(result[i][4]*scale)//2
        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
        cv2.putText(img_cp,result[i][0] + ' : %.2f' % result[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)		
    return img_cp