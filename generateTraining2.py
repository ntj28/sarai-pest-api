from skimage import io
import numpy as np
import csv
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import label2rgb
from skimage.feature import canny
from scipy import ndimage
from skimage import morphology
from skimage.filters import sobel,threshold_otsu, threshold_adaptive
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
import copy
from skimage import measure
from skimage.measure import label
import cv2

#change filename depending on what data you intend to train
species=[]
with open("pests.csv",'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        species.append(row[-1])

#write values to a csv file. filename should be change too, also depending on what data to train
f = open('pesttraining.csv','w')
wr = csv.writer(f)
for s in species:
	for i in xrange(1,15):
		#right path to the folder should be observed
		filename='Training/Pests/'+s+'/'+repr(i)+'.jpg'
		selem = disk(8)

		image = data.imread(filename,as_grey=True)

		thresh = threshold_otsu(image)

		elevation_map = sobel(image)

		markers = np.zeros_like(image)

		if ((image<thresh).sum() > (image>thresh).sum()):
			markers[image < thresh] = 1
			markers[image > thresh] = 2
		else:
			markers[image < thresh] = 2
			markers[image > thresh] = 1


		segmentation = morphology.watershed(elevation_map, markers)

		segmentation = dilation(segmentation-1, selem)
		segmentation = ndimage.binary_fill_holes(segmentation)

		segmentation = np.logical_not(segmentation)
		image[segmentation]=0;

		hist = np.histogram(image.ravel(),256,[0,1])
		
		hist = list(hist[0])
		hist[:] = [float(x) / (sum(hist) - hist[0]) for x in hist]
		hist.pop(0)

		hist.append(s)
		wr.writerow(hist)
		
f.close()
