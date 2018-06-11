# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:49:15 2018

@author: User
"""

import pydicom
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import math
import numpy as np
import time
#import cv2
import copy
from sklearn import preprocessing
import scipy.optimize as opt
import scipy.stats as static
#File=['IM1.dcm']
File=[]
start=time.time()
for i in range(120,121):
    File.append('IM'+str(i))
                                        
#def image_feature(image, image_size):
    #center = 256
    #location_x = np.absolute(np.array([np.arange(image_size).tolist() for i in range(image_size)])-center)/center *(image != 0)
    #location_y = np.absolute(np.array([(i*np.ones(image_size)).tolist() for i in range(image_size)])-center)/center *(image != 0)
    #min_max_scaler  = preprocessing.MinMaxScaler()
    #image = preprocessing.scale(np.array(image))
    #image_x = np.gradient(image)[1] *(image != 0)
    #image_y = np.gradient(image)[0] *(image != 0)
    #image_xx = np.gradient(image_x)[1]
    #image_yy = np.gradient(image_y)[0]
    #image_xy = np.gradient(image_x)[0]
    #curvature_image = preprocessing.scale((image_x**2*image_yy + image_xx*image_y**2 - 2*image_xy)/((image_x**2+image_y**2 + 1e-04)**(3/2)))
    #det = image_xx * image_yy - image_xy**2
    #r = np.sqrt((location_x)**2+(location_y)**2) / (np.max(np.sqrt((location_x)**2+(location_y)**2)))
    #theta = np.arctan2(location_y,location_x) / (np.max(np.arctan2(location_y,location_x))-np.min(np.arctan2(location_y,location_x)))
    
    #return[location_x,location_y,image,r,np.absolute(theta),image_x,image_y]#,curvature_image]
def local_feature(image,add):
    feature = []
    
    
    boundary_y = np.zeros((len(image),add))
    #I_x = np.ga 
    image = np.append(boundary_y,image,axis = 1)
    image = np.append(image,boundary_y,axis = 1)

    boundary_x = np.zeros((add,len(image[0])))
    image = np.append(boundary_x,image,axis = 0)
    image = np.append(image,boundary_x,axis = 0)
    
    for i in range(add,len(image)-add):
        for j in range(add,len(image)-add):
            window = image[i-add:i+add+1,j-add:j+add+1].flatten() 
            
            tmp = []
            tmp.append(np.var(window))
            tmp.append(np.mean(window))
            if np.var(window) == 0:
                tmp.append(0)
                tmp.append(0)
            else:
                tmp.append((np.sum((window - np.mean(window))**3)/np.var(window)**3)/len(window))
                tmp.append((np.sum((window - np.mean(window))**4)/np.var(window)**4)/len(window))
            tmp.append(np.sqrt(np.sum(window**2)))
            tmp.append(2**(-len(window))*(np.sum(window)))
            #feature.append((np.append(np.log10(window+1e-1) ,tmp)))
            feature.append(tmp)
    return np.array(feature)
def Gaussian_Mixture_Model(feature_matrix,cluster): # x,y,I,Ix,Iy,Ixy,Ixx,Iyy,k,det,r,theta
    #feature_5 = local_feature(feature_2)
    
    feature_extra = local_feature(feature_matrix,4)
    print("Done")
    #print(np.shape(feature_5[0]))
    # feature_0 = feature_matrix[0].flatten() #x
    # feature_1 = feature_matrix[1].flatten() #y
    # feature_2 = feature_matrix[2].flatten() #I
    # feature_3 = feature_matrix[3].flatten() #r
    # feature_4 = feature_matrix[4].flatten() #theta
    #feature_5 = feature_matrix[5].flatten() #I_x
    #feature_6 = feature_matrix[6].flatten() #I_y

    # feature_2_train = feature_matrix[2].flatten()[feature_matrix[2].flatten()!=0] #I
    # feature_3_train = feature_matrix[3].flatten()[feature_matrix[2].flatten()!=0] #r
    # feature_4_train = feature_matrix[4].flatten()[feature_matrix[2].flatten()!=0] #theta
    # feature_5_train = feature_matrix[5].flatten()[feature_matrix[2].flatten()!=0]  #I_x
    # feature_6_train = feature_matrix[6].flatten()[feature_matrix[2].flatten()!=0]  #I_y
    
    # feature_7 = feature_matrix[7].flatten()
    # feature_8 = feature_matrix[8].flatten()
    # feature_9 = feature_matrix[9].flatten()
    # feature_10 = feature_matrix[10].flatten()
    # feature_11 = feature_matrix[11].flatten()
    X_train = [feature_extra[i] for i in range(0,len(feature_extra))]
    # X = [np.array([feature_2[i],feature_3[i]]) for i in range(0,len(feature_2))]
    #mean_initial = initail_mean(feature_matrix,cluster)
    gmm_mean = np.transpose(GMM(n_components=cluster).fit(X_train).means_)
    
    #### fmin ####
    
    Y = solve_matrix_lsq(X_train,gmm_mean)
    return [gmm_mean,Y]
    
def solve_matrix_lsq(X,M): #solve X = M*Y
    Y = []
    for i in range(0,len(X)):
        #fun = lambda x : np.linalg.norm(np.multiply(M,x) - np.transpose(X[i]))
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1 })
        # bnds = ((0, None), (0, None))
        x0 = np.transpose(np.zeros(len(X)))
        x0[0] = 1
        bnds = tuple((0,1) for i in range(len(X[0])))
        lb = np.zeros(len(M[0]))
        ub = lb +1
        tmp = opt.lsq_linear(M,X[i],bounds = (lb,ub))
        Y.append(tmp.x)
    return np.transpose(Y)

def atalas(input):
    return([input[0:256,0:256],input[0:256,256:512],input[256:512,0:256],input[256:512,256:512]])
def main(file):
    ds=pydicom.read_file('D:\\MRI_segmentation\\T1 3D\\SE12\\'+file)#+'.dcm')
    #ds=dicom.read_file(file)#+'.dcm')
    
    
    pixel_size=len(ds.pixel_array)
     
    ds_pixel = ds.pixel_array
    #phi = (initial_level_set(pixel_size,390,130,20)) #390 130 20

    #image_feature_data = image_feature(ds_pixel,pixel_size)

    #####################
    # NUMBER OF CLUSTER #
    ##################### 
    CLUSTER = 3

    
    process = Gaussian_Mixture_Model(ds_pixel,CLUSTER)
    Y = process[1]
    print(process[0])
    label_local = []
    
    for i in range(len(Y[0])):
        tmp = (Y[:,i]).tolist() 
        label_local.append(tmp.index(max(tmp)))#+chart_no*CLUSTER)
    label_local = np.reshape(label_local,(512,512))

    for i in range(CLUSTER*0,CLUSTER*1):
        image = ds_pixel*((label_local==i).astype(np.int))
        plt.suptitle("The label" + str(i))
        
        plt.imsave("Cluster" + str(i)+".png",image,cmap = plt.cm.bone,  dpi = 1500)
    




start=time.time()

for i in File:
    main(i)
end=time.time()
print(end - start)