import time
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal
from scipy.linalg import eigh as largest_eigh
from scipy.spatial import distance
from skimage import io
from sklearn import preprocessing


class Training:
	def __init__(self, path):
		self.image_files = [join(path,img) for img in listdir(path) if isfile(join(path,img))]
		self.image_files.sort()
		self.read_images()
		self.pca_covariance2()
		self.find_eigenfaces()
		self.find_weights()
		#self.show_eigenfaces()
		#self.show_face_class()
	
	def read_images(self):
		self.images = []
		self.avg_img = np.zeros_like(io.imread(self.image_files[0]),dtype='int32')
		#print(self.avg_img)
		for file in self.image_files:
			img = io.imread(file)
			#print(img)
			self.avg_img = self.avg_img + img
			self.images.append(img.flatten())
		
		self.avg_img = np.divide(self.avg_img,len(self.image_files))
		self.images = np.array(self.images,dtype='int8')
		flattened_avg = self.avg_img.flatten()
		for ind,val in enumerate(self.images):
			self.images[ind] = np.subtract(self.images[ind],flattened_avg)
		self.images = np.transpose(self.images)
	
	def show_avg_img(self):
		plt.gray()
		plt.axis("off")
		plt.imshow(self.avg_img)
		plt.show()
	
	def pca_covariance(self):
		start = time.time()
		print(self.images)
		print(self.images.shape)
		print(np.transpose(self.images).shape)
		#covariance_mtx = np.matmul(self.images,np.transpose(self.images))
		eigenvalues,eigenvectors = LA.eig(np.cov(self.images))
		print(eigenvalues)
		print(eigenvectors)
		print("Time taken is ",time.time()-start)
	
	def pca_svd(self):
		start = time.time()
		y_mat = np.divide(self.images.T, sqrt(self.images.shape[0]-1))
		U, s, Vh = LA.svd(y_mat)
		print(Vh)
		print("Time taken is ",time.time()-start)
	
	def pca_covariance2(self):
		start = time.time()
		temp = np.transpose(self.images) @ self.images
		self.eigenvalues,self.eigenvectors = LA.eig(temp)
		self.eigenvectors = self.images @ self.eigenvectors
		# print("First")
		# print(self.eigenvectors)
		self.eigenvectors = self.eigenvectors.T
		#print(self.eigenvalues)
		# print("second")
		# print(self.eigenvectors)
		print("Time taken is ",time.time()-start)

	def find_eigenfaces(self):
		eValues = np.array(self.eigenvalues, dtype='int32')
		#print(eValues)
		eFaces = [y for x,y in sorted(zip(eValues,self.eigenvectors),key=lambda x:abs(x[0]))]
		self.eigenfaces = eFaces[-50:]
		self.eigenfaces = np.array(preprocessing.normalize(self.eigenfaces, axis=1, norm='l2'))
		print(self.eigenvectors)
		#print(self.eigenfaces.shape)

	def find_weights(self):
		self.face_class = []
		weights = np.zeros((1,50))
		count = 0
		for img in self.image_files:
			image = io.imread(img)
			weight = self.eigenfaces @ np.transpose(np.subtract(image,self.avg_img).flatten())
			weights = weights + np.array(weight)
			count += 1
			if count == 8:
				self.face_class.append(np.divide(weights,8))
				weights = np.zeros((1,50))
				count = 0

	def show_eigenfaces(self):
		dimX,dimY = self.avg_img.shape
		for img in self.eigenfaces:
			face = np.reshape(img,(dimX,dimY))
			io.imshow(face,cmap='gray')
			plt.show()
	
	def show_face_class(self):
		dimX,dimY = self.avg_img.shape
		for weight in self.face_class:
			img = weight @ self.eigenfaces
			img = np.array(img,dtype='int32')
			img = np.reshape(img,(dimX,dimY))
			img = img + self.avg_img
			io.imshow(img,cmap='gray')
			plt.show()

class Testing:
	def __init__(self,module,img):
		self.eigenfaces = module.eigenfaces
		self.avg_img = module.avg_img
		self.face_class = module.face_class
		self.img = io.imread(img,as_gray=True)
		self.normalized_img = np.transpose(np.subtract(self.img,self.avg_img).flatten())
		self.img.reshape((self.avg_img.shape[0],-1))
		self.projection()
		self.find_match()

	def projection(self):
		self.weights = self.eigenfaces @ self.normalized_img
		print(self.weights)
	
	def find_match(self):
		index = 0
		min_distance = inf
		
		for ind,face in enumerate(self.face_class):
			dist = distance.euclidean(self.weights,face)
			if dist < 10000 and dist < min_distance:
				index = ind
				min_distance = dist
		print(min_distance)
		print(self.face_class[index].shape)
		print(self.eigenfaces.shape)

		matched_face = self.face_class[index] @ self.eigenfaces
		matched_face = np.reshape(matched_face,(self.avg_img.shape))
		matched_face = matched_face + self.avg_img
		self.show_face(self.img,matched_face)
	
	def show_face(self,img1, img2):
		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(img1, cmap='gray')
		plt.subplot(1,2,2)
		plt.imshow(img2, cmap='gray')
		plt.show()



if __name__=="__main__":
	model = Training("training_set")
	model.show_avg_img()
	test = Testing(model,"test_set/subject10.wink")

		

