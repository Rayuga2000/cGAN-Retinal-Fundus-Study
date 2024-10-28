import cv2
import numpy as np
import math
import os

##Reading images
img=cv2.imread("/home/rayuga/Documents/DataSet/GAN/minor_data/256x256_original/240_left.png")

ground=cv2.imread("/home/rayuga/Documents/DataSet/GAN/minor_data/256x256_ground/240_left.png",0)
#size=ground.shape
#print(size)
##cv2.imshow("Ground", ground)

mask=cv2.imread("/home/rayuga/Documents/DataSet/GAN/minor_data/256x256_mask/240_left.png",0)
# mask[mask>127]=255
        

# img=cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,0)
# ground=cv2.copyMakeBorder(ground, 10, 10, 10, 10, cv2.BORDER_CONSTANT,0)
# mask=cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT,0)


##Spliting Channels (0,1,2 represents B,G,R)
#R=img[:,:,2]
G=img[:,:,1]
#B=img[:,:,0]
cv2.imshow("Original Green", G)
cv2.imshow("Ground", ground)
##Thresholding the channels
#R[ground>127]=0
G[ground>127]=0
#B[ground>127]=0

cv2.imshow("Blacked Green", G)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''#Save the images
path = 'Results'

cv2.imwrite(os.path.join(path,"Red.jpg"), R)
cv2.imwrite(os.path.join(path,"Green.jpg"), G)
cv2.imwrite(os.path.join(path,"Blue.jpg"), B)'''

#cv2.imshow("Red", R)
#cv2.imshow("Green", G)
#cv2.imshow("Blue", B)

def get_vessels(black_indices):
  result = []
  sub_list = []
  previous_element = None

  for element in black_indices:
    if previous_element is None or element == previous_element + 1:
      sub_list.append(element)
    else:
      result.append(sub_list)
      sub_list = [element]
    previous_element = element

  # Append the last sub-list to the result list if it is not empty.
  if len(sub_list)!=0:
    result.append(sub_list)

  return result

def generate_channel(c1,c2,x,y):
    for e,row in enumerate(mask):
        white_indices = np.where(row == 255)[0]
        black_indices=[]
        if len(white_indices)!=0:
          for i in white_indices:
              if c1[e][i]==0:
                  black_indices.append(i)
          if len(black_indices)!=0:
            vessels=get_vessels(black_indices)
            #print(e,vessels)
            for vessel in vessels:
                #creating 11x11 matrix on the left side
                matrix=[]
                new_matrix=[]
                col_index=vessel[0]-1
                for i in range(11):
                    for j in range(6):
                        row_index=abs(e-j)#(e-j) if j <= e else -1*(e-j)
                        matrix.append(c1[row_index][col_index])
                        if c1[row_index][col_index]>y:
                            new_matrix.append(c1[row_index][col_index])
                    for j in range(1,6):
                        row_index=e+j
                        matrix.append(c1[row_index][col_index])
                        if c1[row_index][col_index]>y:
                            new_matrix.append(c1[row_index][col_index])
                    col_index-=1

                if len(new_matrix)!=0:
                    p1=sum(new_matrix)//len(new_matrix)
                else:
                    p1=sum(matrix)//len(matrix) #problem
                
                #creating 11x11 matrix on the right side
                matrix=[]
                new_matrix=[]
                col_index=vessel[len(vessel)-1]+1
                for i in range(11):
                    for j in range(6):
                        row_index=abs(e-j)#(e-j) if j <= e else -1*(e-j)
                        matrix.append(c1[row_index][col_index])
                        if c1[row_index][col_index]>y:
                            new_matrix.append(c1[row_index][col_index])
                    for j in range(1,6):
                        row_index=e+j
                        matrix.append(c1[row_index][col_index])
                        if c1[row_index][col_index]>y:
                            new_matrix.append(c1[row_index][col_index])
                    col_index+=1
                    
                if len(new_matrix)!=0:
                    p2=sum(new_matrix)//len(new_matrix)
                else:
                    p2=sum(matrix)//len(matrix) #problem
                
                n=len(vessel)

                b=[]
                for i in range(n):
                    res=((math.comb((n-1),i))*(x**i)*((1-x)**((n-1)-i)))
                    b.append(res)

                diff=abs(p1-p2)
                index=0
                for i in b:
                    c=diff*i
                    if p1>p2:
                        c2[e][vessel[index]]=p1-c
                    else:
                        c2[e][vessel[index]]=p1+c
                    p1=c2[e][vessel[index]]
                    index+=1
            
    return c2

def merge_channel(c1,c2,c,row,col):
    for i in range(row):
        for j in range(col):
            c[i][j]=max(c2[i][j],c1[i][j])
                    
##Blue channel

#horizontal
#b_h2=generate_channel(B.copy(),B.copy(),0,60)

#vertical
#b_v2=generate_channel(B.copy().T,B.copy().T,1,60)
#b_v2=b_v2.T

##Green channel

#horizontal
g_h2=generate_channel(G.copy(),G.copy(),0,60)

#vertical
g_v2=generate_channel(G.copy().T,G.copy().T,1,60)
g_v2=g_v2.T

##Red channel

#horizontal
#r_h2=generate_channel(R.copy(),R.copy(),0,60)

#vertical
#r_v2=generate_channel(R.copy().T,R.copy().T,1,60)
#r_v2=r_v2.T

#get row and col
shape=img.shape
row=shape[0]
col=shape[1]

#create new blue channel
#b=np.empty(shape=[row,col],dtype=np.uint8)
#merge_channel(b_h2,b_v2,b,row,col)

#create new green channel
g=np.empty(shape=[row,col],dtype=np.uint8)
merge_channel(g_h2,g_v2,g,row,col)

#create new red channel
#r=np.empty(shape=[row,col],dtype=np.uint8)
#merge_channel(r_h2,r_v2,r,row,col)

#cv2.imshow("Blue_new", b)
#cv2.imshow("Blue_Horizontal", b_h2)
#cv2.imshow("Blue_Vertical", b_v2)

#cv2.imshow("Green_new", g)
#cv2.imshow("Green_Horizontal", g_h2)
#cv2.imshow("Green_Vertical", g_v2)

#cv2.imshow("Red_new", r)
#cv2.imshow("Red_Horizontal", r_h2)
#cv2.imshow("Red_Vertical", r_v2)
#
#Merge channels and generate new image
#new_image = cv2.merge([b,g,r])

#cv2.imshow("New_Image", new_image)

#Save the images

path = '/home/rayuga/Documents/DataSet/GAN/minor_data/green_new'

'''cv2.imwrite(os.path.join(path,"Red_Horizontal.jpg"), r_h2)
cv2.imwrite(os.path.join(path,"Green_Horizontal.jpg"), g_h2)
cv2.imwrite(os.path.join(path,"Blue_Horizontal.jpg"), b_h2)

cv2.imwrite(os.path.join(path,"Red_Vertical.jpg"), r_v2)
cv2.imwrite(os.path.join(path,"Green_Vertical.jpg"), g_v2)
cv2.imwrite(os.path.join(path,"Blue_Vertical.jpg"), b_v2)

cv2.imwrite(os.path.join(path,"Red_new.jpg"), r)
cv2.imwrite(os.path.join(path,"Green_new.jpg"), g)
cv2.imwrite(os.path.join(path,"Blue_new.jpg"), b)
cv2.imwrite(os.path.join(path,"new_image.jpg"), new_image)'''
cv2.imwrite(os.path.join(path,"Green_new.jpg"), g)


'''
cv2.waitKey(0)
cv2.destroyAllWindows()'''