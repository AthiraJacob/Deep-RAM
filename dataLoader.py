'''
Functions to load and pre-process data before feeding into CNN

Author: Athira
'''
import random
import numpy as np
import pdb
from scipy.ndimage.filters import gaussian_filter
import cv2
import pywt

class DataSet():
	'''
	Helper functions for data
	'''
	def __init__(self, folder, preprocess, wavelet, histEq, tophat, norm, resize, smooth, context):	
		train = np.load(folder + 'data_train.npy').item()
		val = np.load(folder + 'data_val.npy').item()
		test = np.load(folder + 'data_test.npy').item()
		if context is True:
			resize = False
			wavelet = False
		if resize is True:
			train['imgs'] = train['imgs'][:,32:96,32:96]
			val['imgs'] = val['imgs'][:,32:96,32:96]
			test['imgs'] = test['imgs'][:,32:96,32:96]
		if preprocess is True:
			train['imgs'] = self.preprocess(train['imgs'], wavelet,  histEq, tophat, norm, smooth)
			val['imgs'] = self.preprocess(val['imgs'], wavelet, histEq, tophat, norm, smooth)
			test['imgs'] = self.preprocess(test['imgs'], wavelet, histEq, tophat, norm, smooth)
		if context is True:
			train['imgs'] = self.create_context(train['imgs'])
			val['imgs'] = self.create_context(val['imgs'])
			test['imgs'] = self.create_context(test['imgs'])
		self.imgSize = train['imgs'].shape[1]
		self.nTrain = train['area'].shape[0]
		self.nVal = val['area'].shape[0]
		self.nTest = test['area'].shape[0]
		if wavelet is True:
			self.nFeatures = 3
		elif context is True:
			self.nFeatures = 3
		else:
			self.nFeatures = 1
		self.train = train; self.val = val; self.test = test;

	def splitData(self):
		return self.train, self.val, self.test

	def get_side_patches(self, folder, preprocess, wavelet, histEq, tophat, norm, resize):
		train = np.load(folder + 'side_data_train.npy').item()
		val = np.load(folder + 'side_data_val.npy').item()
		test = np.load(folder + 'side_data_test.npy').item()
		if resize is True:
			train['imgs'] = train['imgs'][:,32:96,32:96]
			val['imgs'] = val['imgs'][:,32:96,32:96]
			test['imgs'] = test['imgs'][:,32:96,32:96]
		if preprocess is True:
			train['imgs'] = self.preprocess(train['imgs'], wavelet, histEq, tophat, norm)
			val['imgs'] = self.preprocess(val['imgs'], wavelet, histEq, tophat, norm)
			test['imgs'] = self.preprocess(test['imgs'], wavelet, histEq, tophat, norm)
		self.side_nTrain = train['area'].shape[0]
		self.side_nVal = val['area'].shape[0]
		self.side_nTest = test['area'].shape[0]
		return val, test

	def createOnehot(self,data, percentage):
		temp = data['area']>percentage; #temp = np.stack([temp, 1- temp], axis = 1)
		data['area_onehot'] = temp
		temp = data['dia']>percentage; #temp = np.stack([temp, 1- temp], axis = 1)
		data['dia_onehot'] = temp
		return data

	def divideRandom(self,valRatio = 0.25, testRatio = 0.05):
		'''
		Divide dataset into train and validation set, randomly
		Inputs: valRatio: proportion of data to be allocated as validation
		Outputs: train (dictionary): imgs, area, dia
				 val (dictionary): imgs, area, dia
				 nTrain, nVal: number of examples in each
		'''
		nVal = int(valRatio*self.num)
		nTest = int(testRatio*self.num)
		ind = np.arange(self.num)
		random.shuffle(ind)
		val = {}
		val['imgs'] = np.asarray(self.imgs[ind[0:nVal]],dtype=np.float32)
		val['area'] = self.area[ind[0:nVal]]
		val['dia'] = self.dia[ind[0:nVal]]
		test = {}
		test['imgs'] = np.asarray(self.imgs[ind[nVal:nVal+nTest]],dtype=np.float32)
		test['area'] = self.area[ind[nVal:nVal+nTest]]
		test['dia'] = self.dia[ind[nVal:nVal+nTest]]
		train = {}
		train['imgs'] = np.asarray(self.imgs[ind[nVal+nTest:]],dtype=np.float32)
		train['area'] = self.area[ind[nVal+nTest:]]
		train['dia'] = self.dia[ind[nVal+nTest:]]

		self.nTrain = train['area'].shape[0]
		self.nVal = nVal
		self.nTest = nTest
		return train, val, test, self.nTrain, self.nVal, self.nTest

	def preprocess(self,imgs, wavelet, histEq, tophat = False, norm = '0to1', smooth = False ):

		imgs_new = np.zeros(shape = (imgs.shape))
		n = imgs.shape[0]
		if wavelet is True:
			imgs_new = np.zeros(shape = (n, imgs.shape[1], imgs.shape[2], 3))
			self.nFeatures = 3
			for i in range(n):
				img = imgs[i,:,:]
				img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_LINEAR)
				coeffs = pywt.dwt2(img, 'haar')
				temp = coeffs[0]; temp = (temp - np.min(temp))/(np.max(temp) - np.min(temp) + 0.001); 
				imgs_new[i,:,:,0] = temp
				temp = coeffs[1][0];  temp = (temp - np.min(temp))/(np.max(temp) - np.min(temp)+ 0.001); 
				imgs_new[i,:,:,1] = temp
				temp = coeffs[1][1];  temp = (temp - np.min(temp))/(np.max(temp) - np.min(temp)+ 0.001); 
				imgs_new[i,:,:,2] = temp
		else:
			for i in range(n):
				img = imgs[i,:,:]
				if histEq is True:
					img = image_histogram_equalization(img)
				if tophat is True:
					kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
					blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
					tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
					img = cv2.subtract(img, 2*blackhat)
				if smooth is True:
					img = gaussian_filter(img, 2)
				if norm == '0to1':
					img = img - np.min(img)
					img = img/(np.max(img)+0.001)
				elif norm == 'm1to1':
					img = img - np.min(img)
					img = img/(np.max(img)+0.001)
					img = img - np.mean(img)
				imgs_new[i,:,:] = img

		return imgs_new

	def create_context(self, imgs):
		n = imgs.shape[0]
		self.nFeatures = 3
		imgs_new = np.zeros(shape = (n, 64, 64, self.nFeatures))
		for i in range(n):
			img = imgs[i,:,:]
			imgs_new[i,:,:,0] = img[32:96,32:96]
			imgs_new[i,:,:,1] = cv2.resize(img,None, fx = 0.5,fy = 0.5)
			imgs_new[i,:,:,2] = imgs_new[i,:,:,0]
		return imgs_new





class DataSet_3D():
	'''
	Helper functions for 3D data
	'''
	def __init__(self, folder, preprocess, wavelet, histEq, tophat, norm):	
		data = np.load(folder + 'data_3D.npy').item()
		train = data['train']; val = data['val']; test = data['test'] 
		# pdb.set_trace()
		if preprocess is True:
			train['imgs'] = self.preprocess(train['imgs'], histEq, tophat,  norm)
			val['imgs'] = self.preprocess(val['imgs'], histEq, tophat, norm)
			test['imgs'] = self.preprocess(test['imgs'], histEq, tophat, norm)
		self.imgSize = train['imgs'].shape[1]
		self.nTrain = train['area'].shape[0]
		self.nVal = val['area'].shape[0]
		self.nTest = test['area'].shape[0]
		self.nFeatures = 3
		self.train = train; self.val = val; self.test = test;

	def splitData(self):
		return self.train, self.val, self.test

	def createOnehot(self,data, percentage):
		temp = data['area']>percentage; temp = np.stack([temp, 1- temp], axis = 1)
		data['area_onehot'] = temp
		temp = data['dia']>percentage; temp = np.stack([temp, 1- temp], axis = 1)
		data['dia_onehot'] = temp
		return data


	def preprocess(self,imgs, histEq, tophat, norm = '0to1'):

		imgs_new = np.zeros(shape = (imgs.shape))
		n = imgs.shape[0]
		# pdb.set_trace()
		for i in range(n):
			img = imgs[i,:,:,:]
			if histEq is True:
				img = image_histogram_equalization(img)
			if tophat is True:
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
				blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
			if norm == '0to1':
				img = img - np.min(img)
				img = img/(np.max(img)+0.001)
			elif norm == 'm1to1':
				img = img - np.min(img)
				img = img/(np.max(img)+0.001)
				img = img - np.mean(img)
			imgs_new[i,:,:,:] = img

		return imgs_new
	


def image_histogram_equalization(image, number_bins=256):
	# from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

	# get image histogram
	image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
	cdf = image_histogram.cumsum() # cumulative distribution function
	cdf = 255 * cdf / cdf[-1] # normalize

	# use linear interpolation of cdf to find new pixel values
	image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

	return image_equalized.reshape(image.shape)


