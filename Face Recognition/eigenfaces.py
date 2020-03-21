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
#from sklearn import preprocessing


class Training:
	def __init__(self, path):
		self.image_files = [join(path,img) for img in listdir(path) if isfile(join(path,img))]
		self.image_files.sort()
		self.read_images()
	
	def read_images(self):
		self.images = []
		self.avg_img = np.zeros_like(io.imread(self.image_files[0]),dtype='int32')
		#print(self.avg_img)
		for file in self.image_files:
			img = np.matrix(io.imread(file))
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
		plt.imshow(self.avg_img)
		plt.show()

if __name__=="__main__":
	model = Training("training_set")
	model.show_avg_img()

		

