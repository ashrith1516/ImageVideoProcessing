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

frobenius_norm = []
#Class for creating a training module
class Training:
	def __init__(self, path,eig_filter_no):
		self.image_files = [join(path,img) for img in listdir(path) if isfile(join(path,img))]
		self.image_files.sort()
		self.read_images()
		self.pca_covariance2()
		self.find_eigenfaces(eig_filter_no)
		self.find_weights(eig_filter_no)
		#self.show_eigenfaces()
		self.show_face_class()
	
	#Get normalized images (mean subtracted), flatten and create a matrix of images - A
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
		self.images = np.array(self.images,dtype='int32')
		flattened_avg = self.avg_img.flatten()
		for ind,val in enumerate(self.images):
			self.images[ind] = np.subtract(self.images[ind],flattened_avg)
		self.images = np.transpose(self.images)
	
	def show_avg_img(self):
		plt.gray()
		plt.axis("off")
		plt.title("Average image",fontsize=20)
		plt.imshow(self.avg_img)
		plt.show()
	
	#First method for computing eigenvectors - A * transpose(A)
	def pca_covariance(self):
		start = time.time()
		eigenvalues,eigenvectors = LA.eig(np.cov(self.images))
		print("Time taken is ",time.time()-start)
	
	#Second method for computing eigenvectors - SVD Decomposition
	def pca_svd(self):
		start = time.time()
		y_mat = np.divide(self.images.T, sqrt(self.images.shape[0]-1))
		U, s, Vh = LA.svd(y_mat)
		print("Time taken is ",time.time()-start)
	
	#Third method for computing eigenvectors - transpose(A) * A
	def pca_covariance2(self):
		start = time.time()
		temp = np.transpose(self.images) @ self.images
		self.eigenvalues,self.eigenvectors = LA.eig(temp)
		self.eigenvectors = self.images @ self.eigenvectors
		
		self.eigenvectors = self.eigenvectors.T
		
		print("Time taken is ",time.time()-start)

	#Find top n eigenvectors that become the face space - eigenfaces
	def find_eigenfaces(self,n):
		eValues = np.array(self.eigenvalues, dtype='int32')
		eFaces = [y for x,y in sorted(zip(eValues,self.eigenvectors),key=lambda x:x[0])] 
		self.eigenfaces = eFaces[-n:]
		self.eigenfaces = np.array(preprocessing.normalize(self.eigenfaces, axis=1, norm='l2'))

	#Calculate the face class or the weight vectors for each image
	def find_weights(self,n):
		self.face_class = []
		weights = np.zeros((1,n))
		count = 0
		for img in self.image_files:
			image = io.imread(img)
			weight = self.eigenfaces @ np.transpose(np.subtract(image,self.avg_img).flatten())
			weights = weights + np.array(weight)
			count += 1
			#Considering average weights per person in training set
			if count == 9:
				self.face_class.append(np.divide(weights,9))
				weights = np.zeros((1,n))
				count = 0

	def show_eigenfaces(self):
		dimX,dimY = self.avg_img.shape
		for img in self.eigenfaces:
			face = np.reshape(img,(dimX,dimY))
			plt.axis('off')
			plt.imshow(face,cmap='gray')
			plt.show()
	
	#Projections of each person's weights with the face space or the eigenfaces
	def show_face_class(self):
		dimX,dimY = self.avg_img.shape
		for weight in self.face_class:
			img = np.dot(weight,self.eigenfaces)
			img = np.array(img,dtype='int32')
			img = np.reshape(img,(dimX,dimY))
			img = img + self.avg_img
			plt.axis('off')
			plt.title("Projected face",fontsize=20)
			plt.imshow(img,cmap='gray')
			plt.show()

#Class for test modules
class Testing:
	def __init__(self,module,img,face_thresh=8000,match_thresh=10000):
		self.eigenfaces = module.eigenfaces
		self.avg_img = module.avg_img
		self.face_class = module.face_class
		self.img = io.imread(img,as_gray=True)
		self.normalized_img = np.transpose(np.subtract(self.img,self.avg_img).flatten())
		self.img.reshape((self.avg_img.shape[0],-1))
		self.projection()
		self.face_thresh = face_thresh
		self.match_thresh = match_thresh

	#Projection of input image onto the facespace to get the weights
	def projection(self):
		self.weights = self.eigenfaces @ self.normalized_img
		self.reconstructed_img = self.weights @ self.eigenfaces
		self.reconstructed_img = np.reshape(self.reconstructed_img,(self.avg_img.shape))
		self.reconstructed_img = self.reconstructed_img + self.avg_img
	
	#Function for face recognition
	def find_match(self):
		index = 0

		#Detect if face or not
		projected_face = self.weights @ self.eigenfaces
		face_dist = distance.euclidean(self.normalized_img,projected_face)
		#print(face_dist)
		if face_dist > self.face_thresh:
			self.no_match(self.img,False)
			return
		
		#Detect if match or not
		min_distance = inf
		for ind,face in enumerate(self.face_class):
			dist = distance.euclidean(self.weights,face)
			if dist < self.match_thresh and dist < min_distance:
				index = ind
				min_distance = dist
		
		if min_distance == inf:
			self.no_match(self.img)
			return
		print(min_distance)
		#Display matched face
		matched_face = self.face_class[index] @ self.eigenfaces
		matched_face = np.reshape(matched_face,(self.avg_img.shape))
		matched_face = matched_face + self.avg_img
		self.show_face_match(self.img,matched_face)

	#Display function in case no match is found
	def no_match(self,img1,isFace = True):
		fig = plt.figure()
		if isFace:
			fig.suptitle("Face not recognized",fontsize=20)
		else:
			fig.suptitle("Face not detected",fontsize=20)
		plt.gray()
		plt.axis('off')
		plt.imshow(img1)
		plt.show()
	
	#Display function in case a match is found
	def show_face_match(self,img1, img2):
		fig = plt.figure()
		fig.suptitle("Matched!",fontsize=20)
		plt.subplot(1,2,1)
		plt.axis('off')
		plt.imshow(img1, cmap='gray')
		plt.subplot(1,2,2)
		plt.axis('off')
		plt.imshow(img2, cmap='gray')
		plt.show()

	#Shows difference between Original and Reconstructed images
	def show_diff(self):
		diff_img = self.img - self.reconstructed_img
		fig = plt.figure()
		fig.suptitle("Image differences - Original,Reconstructed,Difference",fontsize=15)
		plt.axis('off')
		plt.subplot(1,3,1)
		plt.imshow(self.img, cmap='gray')
		plt.subplot(1,3,2)
		plt.axis('off')
		plt.imshow(self.reconstructed_img, cmap='gray')
		plt.subplot(1,3,3)
		plt.axis('off')
		plt.imshow(diff_img, cmap='gray')
		plt.show()
		frobenius_diff = LA.norm(diff_img)
		global frobenius_norm
		frobenius_norm.append(frobenius_diff)



if __name__=="__main__":
	model = Training("training_set",100)
	model.show_avg_img()
	test_path = "test_set"
	for img_file in listdir(test_path):
		img_file = join(test_path,img_file)
		test = Testing(model,img_file)
		test.find_match()
	
	# test_path = "NonfaceImages"
	# for img_file in listdir(test_path):
	# 	img_file = join(test_path,img_file)
	# 	test = Testing(model,img_file)
	# 	test.find_match()

	# test_path = "test_set"
	# for img_file in listdir(test_path):
	# 	img_file = join(test_path,img_file)
	# 	test = Testing(model,img_file)
	# 	test.show_diff()
	
	# images = np.array(range(1,31,1))
	# plt.plot(images,np.array(frobenius_norm).reshape(-1,1),'bo')
	# plt.show()

	
	# test_path = "NonfaceImages"
	# for img_file in listdir(test_path):
	# 	img_file = join(test_path,img_file)
	# 	test = Testing(model,img_file)
	# 	test.show_diff()

	# images = np.array(range(1,31,1))
	# plt.plot(images,np.array(frobenius_norm).reshape(-1,1),'bo')
	# plt.show()


		

