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
from matplotlib.colors import LogNorm
from skimage.measure import label

Original_Image=Image.open(sys.argv[1]) # open the image
Grayscale_Image = np.array(Original_Image.convert('L'))  # make sure its greyscale

pixels_to_um = 0.5
pixel_area=pixels_to_um**2 # compute the pixel area from the side length

def plot_spectrum(Grayscale_Image):
   
    # A logarithmic colormap
    plt.imshow(np.abs(Grayscale_Image), norm=LogNorm(vmin=5))
    plt.colorbar()

#plt.figure()
#plot_spectrum(img2)
#plt.title('Fourier transform')

keep_fraction = 0.1

r, c = np.array(Grayscale_Image).shape 

# r*(1-keep_fraction):
Grayscale_Image[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

# Similarly with the columns:
Grayscale_Image[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

#plt.figure()
#plot_spectrum(img2)
#plt.title('Filtered Spectrum')
# Reconstruct the denoised image from the filtered spectrum, keep only the
# real part for display.
#img_new = fftpack.ifft(img2).real # import the library

#plt.figure()
#plt.imshow(img_new, plt.cm.gray)
#plt.title('Reconstructed Image')


img_blur = ndimage.gaussian_filter(Grayscale_Image, 4)

#plt.figure()
#plt.imshow(img_blur, plt.cm.gray)
#plt.title('Blurred image')

plt.show()

bins,values=np.histogram(Grayscale_Image=[0,256], normed=None, weights=None, density=None) # store the histogram results
peaks_and_alleys=np.sign(np.diff(np.sign(np.diff(bins)))) # find the peaks and valleys of the bins
valleys=np.nonzero(peaks_and_alleys==-1)[0]+2 # find where there's are valleys in the histogram (I'm adding 2 because of the dimension change from using diff twice)

threshold=(values[valleys[0]]+np.mean(Grayscale_Image))/2 # try the value of the first valley as our threshold
print(threshold)

#img3 = img2.filter(ImageFilter.FIND_EDGES)
#img3.show()

thresh=np.zeros_like(np.array(Grayscale_Image))
thresh[Grayscale_Image>threshold]=255

#thresh=img2.point(lambda p: p<threshold and 255) # switch to threshold in PIL <------------ switch to threshold in np.array 

thresharr=np.array(thresh) # convert to a numpy array
)

# next I'm going to clean up the image a bit
n_cycles=2 # increase this for more "cleaning"

edgeThresh=n_cycles*3
edge=np.zeros_like(thresh) # mask for detecting if the region is on the edge of the image
edge[:n_cycles,:]=1
edge[-n_cycles:,:]=1
edge[:,:n_cycles]=1
edge[:,-n_cycles:]=1

'''
original data...maybe 1 void and some bits of noise
OOOOOOOOOXXXXOXXXXXXOOOOOOOOOOOOOOOOOXOOOOOOOOOOOOOOOOXOOOOOXOOOOOOOOOOOOOO


first erosion (-1=-1)
OOOOOOOOOoXXoOoXXXXoOOOOOOOOOOOOOOOOOoOOOOOOOOOOOOOOOOoOOOOOoOOOOOOOOOOOOOO

dilate (-1+1=0)
OOOOOOOOOxXXxOxXXXXxOOOOOOOOOOOOOOOOOoOOOOOOOOOOOOOOOOoOOOOOoOOOOOOOOOOOOOO

dilate (-1+1+1=1)
OOOOOOOOxxXXxxxXXXXxxOOOOOOOOOOOOOOOOoOOOOOOOOOOOOOOOOoOOOOOoOOOOOOOOOOOOOO

erode (-1+1+1-1=0)
OOOOOOOOoxXXxxxXXXXxoOOOOOOOOOOOOOOOOoOOOOOOOOOOOOOOOOoOOOOOoOOOOOOOOOOOOOO
'''
thresharr=binary_erosion(thresharr,iterations=n_cycles) # this shrinks all the white objects.  tiny dots will go away
thresharr=binary_dilation(thresharr,iterations=n_cycles*3) # expand objects to try to close up holes
thresharr=binary_erosion(thresharr,iterations=n_cycles*2) # shrink back to the size you started with

def circle_points(resolution, center, radius):   
	"""
    Generate points which define a circle on an image.Centre refers to the centre of the circle
    """   
    radians = np.linspace(0, 2*np.pi, resolution)    c = center[1] + radius*np.cos(radians)#polar co-ordinates 
    r = center[0] + radius*np.sin(radians)
    
    return np.array([c, r]).T# Exclude last point because a closed path should not have duplicate points
#calculates x and y co-ordinates of the points on the periphery of the circle
fig, ax = image_show(Original_Image)
snake = seg.active_contour(Grayscale_Image, points)fig, ax = image_show(Original_Image)
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)#The algorithm then segments particular part of an image by fitting  curve to the edges 
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);

snake = seg.active_contour(Grayscale_Image, points,alpha=0.06,beta=0.3)fig, ax = image_show()
ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);




labels= measure.label(thresharr, connectivity=2, background=0) # label the areas



# this attempts to throw away anything at the edge of the screen so we don't try to measure things that are hanging out of the image bounds
edgeDetect=labels*edge
testRegions=np.unique(labels)[1:]
regions=[]
for R in testRegions:
	if not(np.any(edgeDetect==R)):
		regions.append(R)

areas=[np.count_nonzero(labels==n)*pixel_area for n in regions] # area of each labeled, nonbackground reagion


binary_erosion(Grayscale_image, structure=None, iterations=1, mask=None, output=None, border_value=0, origin=0, brute_force=False) #eroding the image
New_image= (binary_erosion-1)

plt.show()

image_segmented = seg.random_walker(Grayscale_Image,labels)
fig, ax = image_show(image_gray)
ax.imshow(image_segmented == 1, alpha=0.3);
"""
object labeled 1
OOOOOOOOO11111111111OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

eroded object labeled 1
OOOOOOOOOO111111111OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

(object labeled 1)-(eroded object labeled 1)
OOOOOOOOO1OOOOOOOOO1OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
'''


plt.imshow(labels) # start the plot

plt.show() #show the plot



 

