# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:14:56 2020

@author: rpras
"""

import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io, color, measure
from scipy.ndimage import binary_dilation, binary_erosion


img=Image.open(sys.argv[1]) # open the image
img2=img.convert('L') # make sure its greyscale

pixels_to_um = 0.5
pixel_area=pixels_to_um**2 # compute the pixel area from the side length

threshold=100 # might need to pick a value dynamically instead of just arbitrarily choosing a number

thresh=img2.point(lambda p: p<threshold and 255) # switch to threshold in PIL

thresharr=np.array(thresh) # convert to a numpy array

# next I'm going to clean up the image a bit
n_cycles=20 # increase this for more "cleaning"
thresharr=binary_erosion(thresharr,iterations=n_cycles) # this shrinks all the white objects.  tiny dots will go away
thresharr=binary_dilation(thresharr,iterations=n_cycles*3) # expand objects to try to close up holes
thresharr=binary_erosion(thresharr,iterations=n_cycles*2) # shrink back to the size you started with

labels= measure.label(thresharr, connectivity=2, background=0) # label the areas

regions=np.unique(labels)[1:]
areas=[np.count_nonzero(labels==n)*pixel_area for n in regions] # area of each labeled, nonbackground reagion

plt.imshow(labels) # start the plot
centroids=[]
for R in regions: # compute the centroid of each region
	Rx,Ry=np.nonzero(labels==R) 
	centroids.append((np.mean(Ry),np.mean(Rx))) # Y,X order because images are stupid

for i in range(len(regions)):
	plt.annotate("{:3.2f}".format(areas[i]), centroids[i]) # add each annotation

plt.show() #show the plot



#print(areas)
