import cv2
import os

for (root,dirs,_) in os.walk("/home/rayuga/Documents/Project_IEM/Outputs/image_DB"):
    for d in dirs:
        print(d)
        for (_,_,files) in os.walk(root+'/'+d):
            for f in files:
                print(f)
                if f.endswith('.tif'):
                    img = cv2.imread(root+'/'+d+'/'+f)
                    path = '/home/rayuga/Documents/Project_IEM/Outputs/image_DB/original'
                    #os.mkdir(path)

                    cv2.imwrite(os.path.join(path,f"{d}.png"), img)