import cv2
import numpy as np
import os
import fnmatch

path = "./image_DB/"
f_names=[]

for i in range(21,41):
    for root, d_names, f_names in os.walk(path+str(i)):
        f_names=f_names
        root=root
    #Read images
    for f in f_names:
        if fnmatch.fnmatch(f,'*_man*'):
            ground = cv2.imread(root+'/'+f)
        elif fnmatch.fnmatch(f,'*_train*') and f.endswith('.jpg'):
            mask = cv2.imread(root+'/'+f)
        elif f.endswith('.tif'):
            img = cv2.imread(root+'/'+f)
    
    #cv2.imshow("ground",ground)
    #cv2.imshow("mask",mask)
    #cv2.imshow("img",img)
    
    

    bit_or=cv2.bitwise_or(img,ground)
    bit_and=cv2.bitwise_and(bit_or,mask)

    if not os.path.exists("./Input"):
        os.mkdir("./Input")
    cv2.imwrite(os.path.join("./Input",str(i)+'.jpg'),bit_and)
    #cv2.imshow("Result",bit_and)

    """cv2.waitKey(0)
    cv2.destroyAllWindows()"""
