import time
import pickle
from math import fabs, hypot, inf, pi, sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy import ndimage, signal, misc
from scipy.linalg import eigh as largest_eigh
from scipy.spatial import distance
from skimage import io
from skimage.morphology import square

class Codeword:
	def __init__(self,vector,aux):
		self.vector = vector
		self.aux = aux

def colordist(xt,vi):
	xt_sqrd = np.sum(np.array(xt)**2)
	vi_sqrd = np.sum(np.array(vi)**2)
	xtvi_sqrd = np.dot(xt,vi)**2
	p_sqrd = xtvi_sqrd/vi_sqrd
	dist = np.sqrt(xt_sqrd - p_sqrd) if xt_sqrd - p_sqrd >= 0 else 0
	return dist

def brightness(I, aux, alpha, beta):
    Imin = aux[0]
    Imax = aux[1]
    Ilo = Imax * alpha
    Ihi = min(beta*Imax, Imin / alpha)
    if (I >= Ilo and I <= Ihi):
        return True
    return False

def construct_codebook(path,eps1,alpha,beta):
	
	frames = [join(path, img) for img in listdir(path) if isfile(join(path, img))]
	frames.sort()

	code_book = np.zeros_like(io.imread(frames[0],as_gray=True), dtype=list)

	for ind,frame in enumerate(frames):
		t = ind + 1
		print("Processing frame number - ",t)
		img = io.imread(frame)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if t==1:
					code_book[i,j] = []
				R = img[i,j,0]
				G = img[i,j,1]
				B = img[i,j,2]
				# print(R)
				# print(G)
				# print(B)
				xt = [R,G,B]
				I = sqrt(R**2+G**2+B**2)
				match = False
				for code in code_book[i,j]:
					if colordist(xt,code.vector) <= eps1 and brightness(I,code.aux,alpha,beta):
						auxList = code.aux
						fm = auxList[2]
						vm = code.vector
						code.vector = [(fm * vm[0] + xt[0])/(fm + 1), (fm * vm[1] + xt[1])/(fm + 1), (fm * vm[2] + xt[2])/(fm + 1)]
						code.aux = [min(I, auxList[0]), max(I,auxList[1]), fm + 1, max(auxList[3], t - auxList[5]), auxList[4], t]
						match = True
						break
				
				if not match:
					vl = xt
					auxl = [I, I, 1, t - 1, t, t]
					code_book[i,j].append(Codeword(vl,auxl))
				#print(len(code_book[i,j]))

		if t == len(frames):
			for cw in code_book[i, j]:
				auxi = cw.aux
				cw.aux[3] = max(auxi[3], len(frames)-auxi[5]+auxi[4]-1)
	return code_book,len(frames)

def temporal_filtering(path,code_book,num_frames):
	M = num_frames//2
	for i in range(code_book.shape[0]):
		for j in range(code_book.shape[1]):
			filtered_codes = []
			for code in code_book[i,j]:
				if code.aux[3] <= M:
					filtered_codes.append(code)
			code_book[i,j] = filtered_codes
	cwfile = open(path+".code",'wb')
	pickle.dump(code_book,cwfile)
	cwfile.close()
	return code_book

def detect_foreground(image,code_book,eps2,alpha,beta):
	img = io.imread(image)
	output = np.zeros_like(io.imread(image,as_gray=True))

	for i in range(code_book.shape[0]):
		for j in range(code_book.shape[1]):
			R = img[i,j,0]
			G = img[i,j,1]
			B = img[i,j,2]
			xt = [R,G,B]
			I = sqrt(R**2+G**2+B**2)
			match = False
			for code in code_book[i,j]:
				if colordist(xt,code.vector) <= eps1 and brightness(I,code.aux,alpha,beta):
					match = True
					break
			output [i,j] = 0 if match else 255
	return output


if __name__=="__main__":
	alpha = 0.7
	beta = 1.2
	eps1 = 300
	eps2 = 300
	path = "training"
	if isfile(path+".code"):
		cwfile = open(path+".code",'rb')
		code_book = pickle.load(cwfile,encoding='bytes')
		cwfile.close()
	else:
		fat_codebook,num_frames = construct_codebook(path,eps1,alpha,beta)
		code_book = temporal_filtering(path,fat_codebook,num_frames)
	
	background = detect_foreground("testing/PetsD2TeC1_00530.jpg",code_book,eps2,alpha,beta)
	plt.imshow(background,cmap='gray')
	plt.show()