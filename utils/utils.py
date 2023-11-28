# Source: https://github.com/xdshang/real-time-object-detection/blob/master/utils.py

import cv2

def draw_result(img, result, scale=1.0):
    img_cp = img.copy()
    for i in range(len(result)):
        tx = int(result[i][1]*scale)
        ty = int(result[i][2]*scale)
        bx = int(result[i][3]*scale)
        by = int(result[i][4]*scale)
        w = (bx - tx)//2;
        h = (by - ty)//2;

        
        cv2.rectangle(img_cp,(tx,ty),(bx,by),(0,255,0),2)
        cv2.rectangle(img_cp,(tx,ty-20),(bx,ty),(125,125,125),-1)
        cv2.putText(img_cp,result[i][0] + ' : %.2f' % result[i][5],(tx,ty-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)		
    return img_cp