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
import math
from PIL import ImageFilter


img = Image.open (r"C:/Users/rpras/Desktop/missouri s n t/Image Analysis images/Questek/spec 3 10x -1.tiff")  # open the image
img2 = img.convert('L')
 # make sure its greyscale
 

pixels_to_um = 0.5
pixel_area=pixels_to_um**2 # compute the pixel area from the side length

def plot_spectrum(img2):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(img2), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(img2)
plt.title('Fourier transform')

keep_fraction = 0.1

r, c = img2.shape

# r*(1-keep_fraction):
img2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

# Similarly with the columns:
img2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(img2)
plt.title('Filtered Spectrum')
# Reconstruct the denoised image from the filtered spectrum, keep only the
# real part for display.
img_new = fftpack.ifft(img2).real

plt.figure()
plt.imshow(img_new, plt.cm.gray)
plt.title('Reconstructed Image')

from scipy import ndimage
img_blur = ndimage.gaussian_filter(img2, 4)

plt.figure()
plt.imshow(img_blur, plt.cm.gray)
plt.title('Blurred image')

plt.show()

bins,values=np.histogram(img2, bins=50, range=[0,256], normed=None, weights=None, density=None) # store the histogram results
peaks_and_alleys=np.sign(np.diff(np.sign(np.diff(bins)))) # find the peaks and valleys of the bins
valleys=np.nonzero(peaks_and_alleys==-1)[0]+2 # find where there's are valleys in the histogram (I'm adding 2 because of the dimension change from using diff twice)

threshold=(values[valleys[0]]+np.mean(img2))/2 # try the value of the first valley as our threshold
print(threshold)

img3 = img2.filter(ImageFilter.FIND_EDGES)
img3.show()

thresh=img3.point(lambda p: p<threshold and 255) # switch to threshold in PIL

thresharr=np.array(thresh) # convert to a numpy array
erosion_img = img2.filter(ImageFilter.MinFilter(3))
erosion_img.show()

perimeter= (erosion_img-1)
perimeter.show()

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
