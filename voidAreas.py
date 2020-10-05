# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:14:56 2020

@author: rpras
"""

import sys
from PIL import Image , ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io, color, measure
from scipy.ndimage import binary_dilation, binary_erosion
import skimage.segmentation
from PIL import ImageFilter
import math


#img = Image.open (r"C:/Users/rpras/Desktop/missouri s n t/Image Analysis images/Questek/spec 3 10x -1.tiff")  # open the image
img=Image.open(sys.argv[1])
img2 = img.convert('L')
 # make sure its greyscale

pixels_to_um = 0.5
pixel_area=pixels_to_um**2 # compute the pixel area from the side length

bins,values=np.histogram(img2, bins=50, range=[0,256], normed=None, weights=None, density=None) # store the histogram results
peaks_and_alleys=np.sign(np.diff(np.sign(np.diff(bins)))) # find the peaks and valleys of the bins
valleys=np.nonzero(peaks_and_alleys==-1)[0]+2 # find where there's are valleys in the histogram (I'm adding 2 because of the dimension change from using diff twice)

threshold=(values[valleys[0]]+np.mean(img2))/2 # try the value of the first valley as our threshold
print(threshold)
erosion_img = img2.filter(ImageFilter.MinFilter(3)) # i saw you have already written a code for erosion but i wasn't sure which thresharr it would take into consideration
erosion_img.show()

perimeter= erosion_img-[1] # this shows a type error , i belive  that the image is an array and 1 is integer so i tried converting to array but it is  list
perimeter.show()
Roundness =   4A/(pi*Major Axis^2). #finding the roundness of image 
print(Roundness)

img3 = img2.filter(ImageFilter.FIND_EDGES)
#img3.show()

thresh=img2.point(lambda p: p<threshold and 255) # switch to threshold in PIL

thresharr=np.array(thresh) # convert to a numpy array
# next I'm going to clean up the image a bit
n_cycles=2 # increase this for more "cleaning"

edge=np.zeros_like(thresh) # mask for detecting if the region is on the edge
edge[:n_cycles*2,:]=1
edge[-n_cycles*2:,:]=1
edge[:,:n_cycles*2]=1
edge[:,-n_cycles*2:]=1

thresharr=binary_erosion(thresharr,iterations=n_cycles) # this shrinks all the white objects.  tiny dots will go away
thresharr=binary_dilation(thresharr,iterations=n_cycles+1) # expand objects to try to close up holes
thresharr=binary_erosion(thresharr,iterations=1) # shrink back to the size you started with

labels= measure.label(thresharr, connectivity=2, background=0) # label the areas
edgeDetect=labels*edge

testRegions=np.unique(labels)[1:]
regions=[]
for R in testRegions:
	if not(np.any(edgeDetect*(labels==R))):
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
