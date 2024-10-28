import cv2
import os
import numpy as np

for (root,dirs,files) in os.walk("/home/rayuga/Documents/DataSet/GAN/minor_data/optic2"):
    for f in files:
        img=cv2.imread(root+'/'+f,0)
        img[img>127]=255
        cv2.imwrite(os.path.join("/home/rayuga/Documents/DataSet/GAN/minor_data/Enhanced_Optic2/",f),img)
        #print("saving")
