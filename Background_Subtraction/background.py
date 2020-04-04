import pickle
from math import sqrt
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from skimage import io


#Structure for holding a codeword
class Codeword:
	def __init__(self,vector,aux):
		self.vector = vector
		self.aux = aux

#Obtain color distance between a pixel's color and RGB values of a codeword
def colordist(xt,vi):
	xt_sqrd = np.sum(np.array(xt)**2)
	vi_sqrd = np.sum(np.array(vi)**2)
	xtvi_sqrd = np.dot(xt,vi)**2
	p_sqrd = xtvi_sqrd/vi_sqrd
	dist = np.sqrt(xt_sqrd - p_sqrd) if xt_sqrd - p_sqrd >= 0 else 0
	return dist

#Check if brightness change is within limits
def brightness(I, aux, alpha, beta):
    Imin = aux[0]
    Imax = aux[1]
    Ilo = Imax * alpha
    Ihi = min(beta*Imax, Imin / alpha)
    if (I >= Ilo and I <= Ihi):
        return True
    return False

#Construction of the fat codebook
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

#Filter the fat codebook to contain background pixels
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

#Detect foreground pixels of an image
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
				if colordist(xt,code.vector) <= eps2 and brightness(I,code.aux,alpha,beta):
					match = True
					break
			output [i,j] = 0 if match else 255
	return output

#Morpholigical operation - Closing of the Opening
def morph(image):
	opening = ndimage.grey_opening(image,size=(3,3))
	closing = ndimage.grey_closing(image,size=(2,2))
	return closing

#Function to display output images
def display_output(img1,img2,title):
	fig = plt.figure()
	fig.suptitle(title)
	first = fig.add_subplot(1,2,1)
	first.set_title("Original image")
	first.axis('off')
	io.imshow(img1)
	second = fig.add_subplot(1,2,2)
	second.set_title("Foreground image")
	second.axis('off')
	io.imshow(img2,cmap='gray')
	plt.show()

#Function to implement Background subtraction using codebook algorithm
def codebook_bgs(path,image,eps1=300,eps2=300,alpha=0.7,beta=1.2,morphological=True):
	img = io.imread(image)
	if isfile(path+".code"):
		cwfile = open(path+".code",'rb')
		code_book = pickle.load(cwfile,encoding='bytes')
		cwfile.close()
	else:
		fat_codebook,num_frames = construct_codebook(path,eps1,alpha,beta)
		code_book = temporal_filtering(path,fat_codebook,num_frames)

	foreground = detect_foreground(image,code_book,eps2,alpha,beta)
	display_output(img,foreground,"No Morph")
	if morphological:
		foreground = morph(foreground)

	display_output(img,foreground,"Morphed")	



if __name__=="__main__":
	codebook_bgs("training","testing/PetsD2TeC1_00304.jpg")
	codebook_bgs("training","testing/PetsD2TeC1_00420.jpg")
	codebook_bgs("training","testing/PetsD2TeC1_00580.jpg")
	codebook_bgs("training","testing/PetsD2TeC1_00675.jpg")
	codebook_bgs("training","testing/PetsD2TeC1_00750.jpg")

	