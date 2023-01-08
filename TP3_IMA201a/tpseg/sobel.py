#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

from skimage import filters


##############################################

import mrlab as mr

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"

#%% Exercice pratique
# 1.1.1
ima=io.imread('muscle.tif')
sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

ima=io.imread('muscle.tif')
sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.diffGradX(gfima)
grady=mr.diffGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

#1.1.2
ima=io.imread('cell.tif')
plt.imshow(ima, cmap='gray')

sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

sigma=2
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(norme, cmap='gray')

# 1.1.3 - variation du seuil
ima=io.imread('cerveau.tif')
plt.imshow(ima, cmap='gray')

seuilnorme=0.1
sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
contoursnorme =(norme>seuilnorme) 
plt.imshow(255*contoursnorme, cmap='gray')

seuilnorme=0.5 #0.7
sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
contoursnorme =(norme>seuilnorme) 
plt.imshow(255*contoursnorme, cmap='gray')

# 1.2.1 - maxima du gradient
ima=io.imread('cerveau.tif')
plt.imshow(ima, cmap='gray')

sigma=0 # sans filtre
gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
norme=np.sqrt(gradx*gradx+grady*grady)
plt.imshow(255*norme, cmap='gray')

gfima=filters.gaussian(ima,sigma)
gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))
plt.imshow(255*contours, cmap='gray')

seuilnorme=0.1
norme=np.sqrt(gradx*gradx+grady*grady)
contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))
valcontours=(norme>seuilnorme)*contours #pego onde tem max grad
plt.imshow(255*valcontours, cmap='gray')

seuilnorme=0.5
norme=np.sqrt(gradx*gradx+grady*grady)
valcontours=(norme>seuilnorme)
plt.imshow(255*valcontours, cmap='gray')

seuilnorme=1
norme=np.sqrt(gradx*gradx+grady*grady)
valcontours=(norme>seuilnorme)
plt.imshow(255*valcontours, cmap='gray')

#%%


ima=io.imread('cell.tif')
sigma=0
seuilnorme=0.1


gfima=filters.gaussian(ima,sigma)

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')

plt.figure('Image filtrée (passe-bas)')
plt.imshow(gfima, cmap='gray')

gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
      
plt.figure('Gradient horizontal')
plt.imshow(gradx, cmap='gray')

plt.figure('Gradient vertical')
plt.imshow(grady, cmap='gray')

norme=np.sqrt(gradx*gradx+grady*grady)

    
plt.figure('Norme du gradient')
plt.imshow(norme, cmap='gray')

direction=np.arctan2(grady,gradx)
    
plt.figure('Direction du Gradient')
plt.imshow(direction, cmap='gray')


contoursnorme =(norme>seuilnorme) 


plt.figure('Norme seuillée')
plt.imshow(255*contoursnorme)


contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))

plt.figure('Maxima du gradient dans la direction du gradient')
plt.imshow(255*contours)


valcontours=(norme>seuilnorme)*contours
      
plt.figure()
plt.imshow(255*valcontours)
plt.show()

