#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

############## le clear('all') de Matlab
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
################################"

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal
from skimage import filters

from skimage import io
import scipy.ndimage as nd

##############################################

import mrlab as mr

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"

#%% Exercises

# 1.4.1
ima=io.imread('pyramide.tif')
plt.imshow(ima, cmap='gray')
alpha=0.5

gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha)

lpima=gradx2+grady2
plt.imshow(lpima, cmap='gray')

posneg=(lpima>=0)
plt.imshow(255*posneg, cmap='gray')

nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))

for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255
            
plt.imshow(contours, cmap='gray')

log=nd.gaussian_laplace(ima, 2)
plt.imshow(log, cmap='gray')

# 1.5.1
ima=io.imread('pyra-gauss.tif')
plt.imshow(ima, cmap='gray')

sigma=0 # sobel
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

alpha=0.5 #laplacien
gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  
gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha)
lpima=gradx2+grady2
posneg=(lpima>=0)
nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))
for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255          
plt.imshow(contours, cmap='gray')

gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha) #deriche
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

#%%


ima=io.imread('cell.tif')
alpha=0.5

gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

gradx2=mr.dericheGradX(mr.dericheSmoothY(gradx,alpha),alpha)
grady2=mr.dericheGradY(mr.dericheSmoothX(grady,alpha),alpha)  

  

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')


lpima=gradx2+grady2

plt.figure('Laplacien')
plt.imshow(lpima, cmap='gray')


posneg=(lpima>=0)

plt.figure('Laplacien binarisé -/+')
plt.imshow(255*posneg, cmap='gray')

nl,nc=ima.shape
contours=np.uint8(np.zeros((nl,nc)))


for i in range(1,nl):
    for j in range(1,nc):
        if (((i>0) and (posneg[i-1,j] != posneg[i,j])) or
            ((j>0) and (posneg[i,j-1] != posneg[i,j]))):
            contours[i,j]=255
            
   
plt.figure('Contours')
plt.imshow(contours, cmap='gray')
              
#io.imsave('contours.tif',np.uint8(255*valcontours))

  
  
