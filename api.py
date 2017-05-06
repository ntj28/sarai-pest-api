import cv2
import numpy as np
import json
import os
import csv
import copy
import json
from skimage import io
from skimage import data
from skimage import color
from skimage.color import label2rgb
from skimage.feature import canny
from scipy import ndimage
from skimage import morphology
from skimage.filters import sobel,threshold_otsu, threshold_adaptive
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
from skimage import measure
from skimage.measure import label
from flask.ext.cors import CORS, cross_origin
from flask import Flask, request,g
from mahotas.features import haralick

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

global pests
global pestANN 
global diseases
global diseaseANN

def getSpecies(filename):
	species=[]
	with open(filename,'rb') as f:
		reader = csv.reader(f)
		for row in reader:
		    species.append(row[-1])

	return species

def initANN(filename,species,nhidden,step_size,momentum,nsteps,max_err):
	trainingdata=[]
	with open(filename,'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        trainingdata.append(row[:-1])
	inputs = np.empty( (len(trainingdata), len(trainingdata[0])), 'float' )

	for i in range(len(trainingdata)):
	  a = np.array(list(trainingdata[i]))
	  f = a.astype('float')
	  inputs[i,:]=f[:]

	targets= -1 * np.ones( (len(inputs), len(species)), 'float' )

	i=0
	with open(filename,'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        targets[i][species.index(row[-1])]=1
	        i=i+1

	ninputs = len(trainingdata[0])#number of features
	noutput = len(species)#number of classes
	layers = np.array([ninputs, nhidden, noutput])
	
	nnet = cv2.ANN_MLP(layers)

	condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS
	criteria = (condition, nsteps, max_err)
	
	params = dict( term_crit = criteria,train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale = step_size, bp_moment_scale = momentum )
	num_iter = nnet.train(inputs,targets,None,params=params)

	return nnet
	
	
def pestFeatureExtraction(filename):
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

	features = np.empty( (1, len(hist)), 'float' )
	
	a = np.array(list(hist))
	f = a.astype('float')
	features[0,:]=f[:]

	return features

def checkPythonImage(img):

	height, width, depth = img.shape
	angle = 90
	if height > width:
		imageCenter = tuple(np.array(img.shape)/2)
		img = ndimage.rotate(img, 90)
	 
	baseheight = 840
	hpercent = (baseheight / float(width))
	wsize = int((float(height) * float(hpercent)))
	resizeImg = cv2.resize(img, (800, 640))

	return resizeImg

def checkPythonGrayImage(img):

	height, width = img.shape
	angle = 90
	if height > width:
		imageCenter = tuple(np.array(img.shape)/2)
		img = ndimage.rotate(img, 90)
	 
	baseheight = 840
	hpercent = (baseheight / float(width))
	wsize = int((float(height) * float(hpercent)))
	resizeImg = cv2.resize(img, (800, 640))

	return resizeImg

def diseaseFeatureExtraction(filename):
	selem = disk(8)

	#THRESHOLDING STUFFFFFFFFFFFFFFFFFFFFFFFFFFFF
	image = data.imread(filename)
	image = checkPythonImage(image)

	hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grayimage = checkPythonGrayImage(grayimage)

	thresh = threshold_otsu(grayimage)

	elevation_map = sobel(grayimage)

	markers = np.zeros_like(grayimage)

	if ((grayimage<thresh).sum() > (grayimage>thresh).sum()):
		markers[grayimage < thresh] = 1
		markers[grayimage > thresh] = 2
	else:
		markers[grayimage < thresh] = 2
		markers[grayimage > thresh] = 1


	segmentation = morphology.watershed(elevation_map, markers)

	segmentation = dilation(segmentation-1, selem)
	segmentation = ndimage.binary_fill_holes(segmentation)

	segmentation = np.logical_not(segmentation)
	grayimage[segmentation]=0;

	watershed_mask = np.empty_like(grayimage, np.uint8)
	width = 0
	height = 0
	while width < len(watershed_mask):

		while height < len(watershed_mask[width]):

			if grayimage[width][height] == 0:
				watershed_mask[width][height] = 0
			else:
				watershed_mask[width][height] = 1

			height += 1
			pass

		width += 1
		height = 0
		pass



	#SPLITTING STUFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
	image = cv2.bitwise_and(image,image,mask = watershed_mask)
	hsv = ''
	if image.shape[2] == 3:
		hsv = color.rgb2hsv(image)
	elif image.shape[2] == 4:
		image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
		hsv = color.rgb2hsv(image)
	h,s,v = cv2.split(hsv2)

	#MASKING STUFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
	mask = cv2.inRange(h, 40, 80)
	cv2.bitwise_not(mask, mask)

	res = cv2.bitwise_and(image,image, mask= mask)
	res_gray = cv2.bitwise_and(grayimage,grayimage, mask=mask)
	
	harfeatures = haralick(res.astype(int), ignore_zeros=True, return_mean=True)

	#glcm = greycomatrix(res_gray, [5], [0], 256)
	#contrast = greycoprops(glcm, 'contrast')[0, 0]
	#ASM = greycoprops(glcm, 'ASM')[0, 0]
	#dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
	#homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
	#energy = greycoprops(glcm, 'energy')[0, 0]
	
	features = []
	
	#features.append(contrast)
	#features.append(ASM) 
	#features.append(dissimilarity)
	#features.append(homogeneity)
	#features.append(energy)

	hist = cv2.calcHist([res],[0],None,[256],[0,256])
	w, h, c = res.shape
	numPixel = w * h

	num = 0
	for index in hist:

		if num != 0 and num<255:
			features.append(index[0]/(numPixel-hist[0][0]))

		num = num + 1

		pass

	for harfeature in harfeatures:
		features.append(harfeature)
		pass	

	output = np.empty( (1, len(features)), 'float' )
	
	a = np.array(list(features))
	output[0,:]=a[:]
	return output

def predict(nnet,species,features):
	prediction = np.empty(shape=(1,len(species)))
	nnet.predict(features,prediction)

	return prediction[0]


@app.route("/pestImageSearch", methods=['POST', 'OPTIONS'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def pestImageSearch():
	filename = request.form.get("filename","")

	pestFeatures = pestFeatureExtraction(filename)

	pestPrediction = predict(pestANN,pests,pestFeatures)
	
	sortedPestPrediction=np.argsort(pestPrediction)[::-1]
	pestData=[]
	for i in xrange(0,5):
		pestData.append({'name':pests[sortedPestPrediction[i]],'confidence':pestPrediction[sortedPestPrediction[i]]})

	#result = {'data': pestData, 'features': pestFeatures[0].tolist() }
	result = {'data': pestData}

	return json.dumps(result)

@app.route("/diseaseImageSearch",methods=["POST"])
def diseaseImageSearch():
	filename = request.form.get("filename","")

	diseaseFeatures = diseaseFeatureExtraction(filename)

	diseasePrediction = predict(diseaseANN,diseases,diseaseFeatures)
	
	sortedDiseasePrediction=np.argsort(diseasePrediction)[::-1]
	diseaseData=[]
	for i in xrange(0,5):
		diseaseData.append({'name':diseases[sortedDiseasePrediction[i]],'confidence':diseasePrediction[sortedDiseasePrediction[i]]})
	
	#result = {'data': diseaseData,'features':diseaseFeatures[0].tolist()}

	result = {'data': diseaseData}

	return json.dumps(result)

@app.route("/addTrainingData",methods=["POST"])
def addTrainingData():	
	flag = request.form.get("flag","")
	filename = request.form.get("filename","")
	
	if flag != "true":
		os.remove(filename)
	else:
		classification = request.form.get("type","")
		target = request.form.get("target","")
		
		DIR = "pending/"+classification+"/"+target
		if not os.path.exists(DIR):
			os.makedirs(DIR)
		index = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		
		os.rename(filename,DIR+"/"+str(index)+".jpg")

	return "HAHAHAHAHAHA"

# @app.route("/pestAddTrainingData",methods=["POST"])
# def pestAddTrainingData():
# 	features = request.form.get("input","")
# 	target = request.form.get("target","")
# 	features = features.split(",")
# 	features = map(float, features)
	
# 	inputs = np.empty( (1, len(features)), 'float' )
# 	a = np.array(list(features))
# 	f = a.astype('float')
# 	inputs[0,:]=f[:]

# 	targets= -1 * np.ones( (1, len(pests)), 'float' )

# 	targets[0][pests.index(target)]=1

# 	condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS
# 	criteria = (condition, 900, 0.0000000001)
	
# 	params = dict( term_crit = criteria,train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale = 0.1, bp_moment_scale = 0.003 )

# 	#pestANN.train(inputs,targets,None,params=params,flags=cv2.ANN_MLP_UPDATE_WEIGHTS)

# 	return "HAHA"

# @app.route("/diseaseAddTrainingData",methods=["POST"])
# def diseaseAddTrainingData():
# 	features = request.form.get("input","")
# 	target = request.form.get("target","")
# 	features = features.split(",")
# 	features = map(float, features)
	
# 	inputs = np.empty( (1, len(features)), 'float' )
# 	a = np.array(list(features))
# 	f = a.astype('float')
# 	inputs[0,:]=f[:]

# 	targets= -1 * np.ones( (1, len(diseases)), 'float' )

# 	targets[0][diseases.index(target)]=1

# 	condition = cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS
# 	criteria = (condition, 50000, 0.0000000001)
	
# 	params = dict( term_crit = criteria,train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,bp_dw_scale = 0.08, bp_moment_scale = 0.003 )

# 	#diseaseANN.train(inputs,targets,None,params=params,flags=cv2.ANN_MLP_UPDATE_WEIGHTS)

# 	return "HAHA"

	

if __name__ == "__main__":
	pests = getSpecies('pests.csv')
	pestANN = initANN('pesttraining.csv',pests, 90, 0.003, 0.1, 900, 0.0000000001)
	#diseases = getSpecies('diseases.csv')
	#diseaseANN = initANN('diseasetraining.csv',diseases, 64, 0.003, 0.08, 50000, 0.0000000001)
	#app.run(debug=True, host = '192.168.0.12')
	app.run(debug=True, host = '127.0.0.1')
	
