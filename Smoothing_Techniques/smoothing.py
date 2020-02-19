from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from math import sqrt,pi

# Manual mean filter to show the working of convolution
# Takes in the size of the kernel and input image as parameters
def mean_filter_manual(size,img):
	correction = size//2
	kernel = np.divide(np.ones((size,size)),size*size)
	res = np.copy(img)

	for i in range(correction,len(img)-correction):
		for j in range(correction,len(img[0])-correction):
			sum = 0

			subImg = img[i-correction:i+correction+1,j-correction:j+correction+1]

			for x in range(size):
				for y in range(size):
					sum += kernel[x][y]*subImg[x][y]
			res[i-1][j-1] = int(sum)

	return res

# Mean filter using the convolution function from ndimage which is faster
# Takes in the size of the kernel and input image as parameters
def mean_filter_auto(size,img):
	kernel = np.divide(np.ones((size,size)),size*size)
	res = ndimage.convolve(img,kernel)
	return res

# Gaussian function
# Takes in the step, mean and standard deviation as parameters
def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

# Function to create a gaussian kernel
# Size of the kernel is determined by the standard deviation value - Size = 5 * standard deviation
# Takes in the standard deviation as parameter
def gaussian_kernel(sigma):
	size = int(5 * sigma)
	size = size+1 if size%2 == 0 else size
	limit = size//2
	#The range we pick values from must be equal to 5 times sigma
	gaus_range = [x for x in range(-limit,limit+1)]
	kernel = [gaussian(x,0,sigma) for x in gaus_range]
	normalized_kernel = [x/kernel[limit] for x in kernel]
	print(normalized_kernel)
	total = sum(normalized_kernel)
	gauss_kernel = [x/total for x in normalized_kernel]
	print(gauss_kernel)
	return np.array(gauss_kernel).reshape(1,size)

# Applies a gaussian filter to an image and returns the filtered image
# Takes in standard deviation and input image as parameters
def gaussian_filter(sigma,img):
	kernel = gaussian_kernel(sigma)
	print(kernel)
	res = ndimage.convolve(img,kernel)
	kernel = np.transpose(kernel)
	res = ndimage.convolve(img,kernel)
	return res

# Applies a median filter with manual convolution and returns the filtered image
# Takes in the size of kernel and input image as parameters
def median_filter_manual(size,img):
	correction = size//2
	kernel = np.zeros((size,size))
	res = np.copy(img)
	for i in range(correction,len(img)-correction):
		for j in range(correction,len(img[0])-correction):
			kernel = img[i-correction:i+correction+1,j-correction:j+correction+1]
			res[i,j] = np.median(kernel)
	
	return res

# Applies a median filter using ndimage library which is much faster and returns the filtered image
# Takes in the size of the kernel and input image as parameters
def median_filter_auto(sz,img):
	res = ndimage.median_filter(img,size = sz)
	return res


# Function to display the results of filtering on 4x4 subplots
# Takes in 2 raw images, 2 filtered images, type of filter applied and messages for filtered images as parameters
def display_output(img1,img2,img3,img4,filter,msg):
	fig = plt.figure()
	first = fig.add_subplot(2,2,1)
	io.imshow(img1)
	first.set_title("Gaussian noise image")
	first.axis('off')
	second = fig.add_subplot(2,2,2)
	io.imshow(img2)
	second.set_title("Impulsive noise image")
	second.axis('off')
	a = fig.add_subplot(2,2,3)
	io.imshow(img3)
	a.set_title(filter + " filter with " + msg)
	a.axis('off')
	b = fig.add_subplot(2,2,4)
	io.imshow(img4)
	b.set_title(filter + " filter with " + msg)
	b.axis('off')
	plt.show()

    

#Tests
if __name__ == "__main__":
	img1 = io.imread("NoisyImage1.jpg")
	img2 = io.imread('NoisyImage2.jpg')

	mean_res1 = mean_filter_auto(3,img1)
	mean_res2 = mean_filter_auto(3,img2)
	display_output(img1,img2,mean_res1,mean_res2,"Mean","size 3")

	mean_res1 = mean_filter_auto(5,img1)
	mean_res2 = mean_filter_auto(5,img2)
	display_output(img1,img2,mean_res1,mean_res2,"Mean","size 5")

	gaussian_res1 = gaussian_filter(1.4,img1)
	gaussian_res2 = gaussian_filter(1.4,img2)
	display_output(img1,img2,gaussian_res1,gaussian_res2,"Gaussian","sigma 1.4")

	gaussian_res1 = gaussian_filter(1.8,img1)
	gaussian_res2 = gaussian_filter(1.8,img2)
	display_output(img1,img2,gaussian_res1,gaussian_res2,"Gaussian","sigma 1.8")

	median_res1 = median_filter_auto(3,img1)
	median_res2 = median_filter_auto(3,img2)
	display_output(img1,img2,median_res1,median_res2,"Median","size 3")

	median_res1 = median_filter_auto(5,img1)
	median_res2 = median_filter_auto(5,img2)
	display_output(img1,img2,median_res1,median_res2,"Median","size 5")




