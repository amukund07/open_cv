import numpy as np
import cv2

img= cv2.imread('assets/logo.png')
img= cv2.resize(img,(0,0),fx=0.75,fy=0.75)
grey= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners= cv2.goodFeaturesToTrack(grey,100,0.01,10)
corners= np.int8(corners)

for corner in corners:
    x,y= corner.ravel()
    cv2.circle(img,(x,y),3,[225,0,5])



cv2.imshow("ing",img)
cv2.waitKey(0)
cv2.destroyAllWindows()