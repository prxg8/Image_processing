# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:14:56 2020

@author: rpras
"""

import sys
from PIL import Image, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io, color, measure
from scipy.ndimage import binary_dilation, binary_erosion
import cv2 as cv


img=Image.open(sys.argv[1]) # open the image
img2=img.convert('L') # make sure its greyscale
img2=img2.filter(ImageFilter.GaussianBlur(radius=10)) # blur radius to get rid of image scratches

pixels_to_um = 0.5
pixel_area=pixels_to_um**2 # compute the pixel area from the side length

bins,values=np.histogram(img2, bins=50, range=[0,256], normed=None, weights=None, density=None) # store the histogram results
peaks_and_alleys=np.sign(np.diff(np.sign(np.diff(bins)))) # find the peaks and valleys of the bins
valleys=np.nonzero(peaks_and_alleys==-1)[0]+2 # find where there's are valleys in the histogram (I'm adding 2 because of the dimension change from using diff twice)

threshold=(values[valleys[0]]+np.mean(img2))/2 # try the value of the first valley as our threshold
print(threshold)

edges = cv.Canny(img2,min,max_thresh=threshold,kernel=3)# min thresh, max thresh values given and size of kernel is (3,3)
plt.imshow(edges,cmap = 'gray')

thresh=img2.point(lambda p: p<threshold and 255) # switch to threshold in PIL

thresharr=np.array(thresh) # convert to a numpy array

# next I'm going to clean up the image a bit
n_cycles=2 # increase this for more "cleaning"

edgeThresh=n_cycles*3
edge=np.zeros_like(thresh) # mask for detecting if the region is on the edge
edge[:n_cycles,:]=1
edge[-n_cycles:,:]=1
edge[:,:n_cycles]=1
edge[:,-n_cycles:]=1


thresharr=binary_erosion(thresharr,iterations=n_cycles) # this shrinks all the white objects.  tiny dots will go away
thresharr=binary_dilation(thresharr,iterations=n_cycles*3) # expand objects to try to close up holes
thresharr=binary_erosion(thresharr,iterations=n_cycles*2) # shrink back to the size you started with

labels= measure.label(thresharr, connectivity=2, background=0) # label the areas
edgeDetect=labels*edge

testRegions=np.unique(labels)[1:]
regions=[]
for R in testRegions:
	if not(np.any(edgeDetect==R)):
		regions.append(R)

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
