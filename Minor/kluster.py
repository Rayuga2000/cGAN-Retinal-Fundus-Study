from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os

#feature extraction
def extractFeature(matrix):
    features = []

    for i,row in enumerate(mask):
        white_indices = np.where(row == 255)[0]
        if len(white_indices)!=0:
            for col in white_indices:
                sub = []
                for x in range(i - 1, i + 2):
                    for y in range(col - 1, col + 2):
                        if not(x==i and y==col):
                            substract = abs(int(matrix[x][y]) - int(matrix[i][col]))
                            sub.append(substract)

                minimum = min(sub)
                features.append([matrix[i][col], minimum])
    return features

matrix=cv2.imread("Results/remove_vessel/24/Green_new.jpg",0)
mask=cv2.imread("C:/Users/ghosh/Documents/GitHub/Project_IEM/image_DB/24/24_training_mask-0000.jpg",0)

features = extractFeature(matrix)

# K-means clustering

kmeans = KMeans(n_clusters=3,init="k-means++",n_init="auto",random_state=0)

kmeans.fit_predict(features)

labels = kmeans.labels_
#print(np.unique(labels))

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(19,6))
plt.scatter(*zip(*features), c=labels, cmap='viridis')

plt.title('K-Means Clustering')
plt.show()

#get row and col
shape=matrix.shape
row=shape[0]
col=shape[1]

k0_img=np.zeros((row,col),dtype=np.uint8)
k1_img=np.zeros((row,col),dtype=np.uint8)
k2_img=np.zeros((row,col),dtype=np.uint8)

#region creation
for i,row in enumerate(mask):
    white_indices = np.where(row == 255)[0]
    if len(white_indices)!=0:
        for col in white_indices:
            sub = []
            for x in range(i - 1, i + 2):
                for y in range(col - 1, col + 2):
                    if not(x==i and y==col):
                        substract = abs(int(matrix[x][y]) - int(matrix[i][col]))
                        sub.append(substract)

            minimum = min(sub)
            f_index=features.index([matrix[i][col],minimum])
            #print(labels[f_index])
            match labels[f_index]:
                case 0:
                    k0_img[i][col]=255
                    #print("kluster 1")
                case 1:
                    k1_img[i][col]=255
                    #print("kluster 2")
        
                case 2:
                    k2_img[i][col]=255
                    #print("kluster 3")
        
                case _:
                    print("Error!!!")

#show images
'''cv2.imshow("K0", k0_img)
cv2.imshow("K1", k1_img)
cv2.imshow("K2", k2_img)'''

#Save the images

path = 'Results/kluster/24'

cv2.imwrite(os.path.join(path,"k0.jpg"), k0_img)
cv2.imwrite(os.path.join(path,"k1.jpg"), k1_img)
cv2.imwrite(os.path.join(path,"k2.jpg"), k2_img)

cv2.waitKey(0)
cv2.destroyAllWindows()