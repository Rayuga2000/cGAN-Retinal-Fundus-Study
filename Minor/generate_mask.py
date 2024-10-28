import cv2
import os
import numpy as np

for (root,dirs,files) in os.walk("/home/rayuga/Documents/Linux_Backup/DataSet/GAN/Training_Data-(1024x1024)/Test/1024x1024"):
    for f in files:
        mask=np.zeros(shape=[1024,1024],dtype=np.uint8)
        img=cv2.imread(root+'/'+f)
        mask[img[:,:,2]>50]=255
        cv2.imwrite(os.path.join("/home/rayuga/Documents/Linux_Backup/DataSet/GAN/mask_1024x1024",f),mask)
        #print("saving")
