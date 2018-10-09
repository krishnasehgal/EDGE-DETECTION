
# coding: utf-8

# # EDGE DETECTION

# In[ ]:


import cv2
import numpy as np
import sys


# In[2]:


img= cv2.imread('/Users/krishna/Downloads/proj1_cse573-5/task1.png', 0)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
value=(img)
#print(value)


# In[3]:


soble_x1=[[1,0,-1],[2,0,-2],[1,0,-1]]
soble_x=np.array(soble_x1)
soble_y1=[[1,2,1],[0,0,0],[-1,-2,-1]]
soble_y=np.array(soble_y1)


# In[4]:


def ip(values):
        row, col= values.shape
        #print(row,col)
        
        padding=np.array([[0 for i in range(col+2)] for j in range(row+2)])  
        
        for x in range(1, row+1): 
            for y in range(1, col+1):
                padding[x, y]=values[x-1, y-1]
        return(padding)


# In[5]:


def func(image,kernel):
    row= image.shape[0]
    col= image.shape[1]
    max=0
    #print(row, col)
    edged=np.array([[0 for i in range(col)] for j in range(row)]) 
    #edged = np.zeros(value.shape)
    for i in range(1, row-1):
        for j in range(1,col-1):
            z=kernel[0][0]*image[i-1][j-1]+kernel[0][1]*image[i-1][j]+kernel[0][2]*image[i-1][j+1]+kernel[1][0]*image[i][j-1]+kernel[1][1]*image[i][j]+kernel[1][2]*image[i][j+1]+kernel[2][0]*image[i+1][j-1]+kernel[2][1]*image[i+1][j]+kernel[2][2]*image[i+1][j+1]
            edged[i][j]=z
            
    return(edged)


# In[6]:


def normalise(matrix):
    maximum=0
    minimum=matrix[1][1]
    pos_edge_x =[[0 for x in range(len(matrix[0]))] for y in range(len(matrix))] 
    
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]>maximum:
                maximum=matrix[i][j]

    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]<minimum:      
                minimum=matrix[i][j]
                
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            pos_edge_x[i][j] = ((matrix[i][j] - minimum) / (maximum - minimum))
    

    return(pos_edge_x)


# In[7]:


pad=np.array(ip(value))


# In[8]:


max_min=np.array(func(pad,soble_x))
xedge=np.array(normalise(max_min))
#cv2.namedWindow('pos_edge_x_dir', cv2.WINDOW_NORMAL)
#cv2.imshow('pos_edge_x_dir', xedge)
#cv2.waitKey(0)


# In[ ]:


max_min=np.array(func(pad,soble_y))
yedge=np.array(normalise(max_min))
cv2.namedWindow('pos_edge_y_dir', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', yedge)
cv2.waitKey(0)
cv2.destroyAllWindows()

