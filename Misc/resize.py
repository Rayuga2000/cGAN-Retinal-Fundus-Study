import cv2
import os

for (root,dirs,files) in os.walk("/home/rayuga/Downloads/icons/"):
    for f in files:
        img=cv2.imread(root+f)
        img=cv2.resize(img,(32,32))
        cv2.imwrite(os.path.join("/home/rayuga/Documents/GitHub/Project/Portfolio/Theme1/assets/",f),img)
        