import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from math import sqrt,pi,degrees,atan
import cv2



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
	total = sum(normalized_kernel)
	gauss_kernel = [x/total for x in normalized_kernel]

	return np.array(gauss_kernel).reshape(1,size)

# Applies a gaussian filter to an image and returns the filtered image
# Takes in standard deviation and input image as parameters
def gaussian_filter(img,sigma):
	kernel = gaussian_kernel(sigma)
	#print(kernel)
	res = ndimage.convolve(img,kernel)
	kernel = np.transpose(kernel)
	res = ndimage.convolve(img,kernel)
	return res

#Obtains the neigboring matrix element position based on the edge normal value
#Takes in the edge normal value in degrees and outputs the neigboring pixel positions
def positions(num):
	if 0 <= num < 22.5 or 157.5 <= num <= 180:
		return ((0,-1),(0,1))
	elif 22.5 <= num < 67.5:
		return ((1,-1),(-1,1))
	elif 67.5 <= num < 112.5:
		return ((1,0),(-1,0))
	else:
		return ((1,1),(-1,-1))

#Performs image smoothing and creates gradient, strength and orientation grids
#Takes in the image and sigma value for smoothing as parameters and returns edge strength and orientation images
def canny_enhancer(img, sigma):
	#Gaussian smoothing
	smooth_img = gaussian_filter(img,sigma)
	kernel_jx = [[-1,0,1]]
	kernel_jy = [[-1],[0],[1]]
	#Gradients on x and y co-ordinates
	jx = signal.convolve2d(smooth_img,kernel_jx,boundary='fill',mode='same')
	jy = signal.convolve2d(smooth_img,kernel_jy,boundary='fill',mode='same')

	#Edge strength
	es = np.hypot(jx, jy)
	#Edge orientation
	eo = np.degrees(np.arctan2(jy, jx))

	return np.array(es,dtype=np.int32),np.array(eo,dtype=np.int32)

#Performs thinning of edges to 1 pixel
#Takes in the strength and orientation matrices and returns the strength matrix with 1 pixel edges
def nonmax_suppression(strength,orientation):
	intensity = np.copy(strength)
	for i in range(1,len(intensity)-1):
		for j in range(1,len(intensity[0])-1):
			directions = positions(orientation[i][j])
			x1 = directions[0][0]
			y1 = directions[0][1]
			x2 = directions[1][0]
			y2 = directions[1][1]
			neighbor1 = strength[i+x1][j+y1]
			neighbor2 = strength[i+x2][j+y2]
			if strength[i][j] < neighbor1 or strength[i][j] < neighbor2:
				intensity[i][j] = 0
			else:
				intensity[i][j] = strength[i][j]
	
	return intensity

#Performs edge tracking
#Takes in the image, starting location of pixel, orientation matrix, low threshold and a visited set for dfs
#Performs edge linking traversing in the direction parallel to edge normal
def chaining(img,i,j,ori,low,visited):
	visited.add((i,j))
	#90 is added to get the perpendicular value
	directions = positions((ori[i][j]+90)%180)
	x1 = directions[0][0]
	y1 = directions[1][1]
	x2 = directions[1][0]
	y2 = directions[1][1]
	while (x1,y1) not in visited and x1 >= 0 and x1 < len(img) and y1 >=0 and y1 < len(img[0]) and img[x1][y1] >= low:
		img[x1][y1] = 255
		chaining(img,x1,y1,ori,low,visited)
	while (x2,y2) not in visited and x2 >= 0 and x2 < len(img) and y2 >=0 and y2 < len(img[0]) and img[x2][y2] >= low:
		img[x2][y2] = 255
		chaining(img,x2,y2,ori,low,visited)


#Performs hysteresis thresh
#Takes in the image, orientation matrix, low and high threshold values as parameters
def hysteresis_threshold(img,ori,low, high):
	for i in range(len(img)):
		for j in range(len(img[0])):
			visited = set()
			if img[i][j] >= high:
				img[i][j] = 255
				chaining(img,i,j,ori,low,visited)
			

#Inverting black and white values
def invert(img):
	inv = [[255]*len(img[0])]*len(img)
	res = np.subtract(inv,img)
	return np.array(res,dtype = np.int32)

#Displaying result images on a plot
def display_output(img1,img2):
	fig = plt.figure()
	first = fig.add_subplot(1,2,1)
	plt.gray()
	plt.imshow(img1,vmin=0,vmax=255)
	first.set_title("Original Image")
	first.axis('off')
	second = fig.add_subplot(1,2,2)
	second.set_title("Edge image")
	second.axis('off')
	plt.gray()
	plt.imshow(img2,vmin=0,vmax=255)
	plt.show()

#Canny edge detection process performing the canny enhancer, nonmax suppression and hysteresis threshold on an image
def canny_edge_detector(img,sigma=3,low=5,high=20):
	strength,orientation = canny_enhancer(img,sigma)
	suppressed = nonmax_suppression(strength,orientation)
	hysteresis_threshold(suppressed,orientation,low,high)
	res = invert(suppressed)
	display_output(img,res)
			

if __name__ == "__main__":
	
	img = cv2.imread("Flowers.jpg",0)
	canny_edge_detector(img,1,10,20)
	img1 = cv2.imread("Syracuse_01.jpg",0)
	canny_edge_detector(img1,1,10,20)
	img2 = cv2.imread("Syracuse_02.jpg",0)
	# canny_edge_detector(img2,1,5,20)
	# canny_edge_detector(img2,2,5,20)
	# canny_edge_detector(img2,3,5,20)
	# canny_edge_detector(img2,1,20,40)
	canny_edge_detector(img2,1,10,20)
	img3 = cv2.imread("seattle.jpg",0)
	canny_edge_detector(img3,1,10,20)
	

	
