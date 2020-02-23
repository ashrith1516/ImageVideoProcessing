import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from math import sqrt,pi,degrees,atan
from skimage.feature import corner_harris,corner_peaks
from matplotlib.patches import Circle
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
	offset = size//2
	#The range we pick values from must be equal to 5 times sigma
	gaus_range = [x for x in range(-offset,offset+1)]
	kernel = [gaussian(x,0,sigma) for x in gaus_range]
	normalized_kernel = [x/kernel[offset] for x in kernel]
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

def gradients(img):
	kernel_jx = [[-1,0,1]]
	kernel_jy = [[-1],[0],[1]]
	#Gradients on x and y co-ordinates
	jx = signal.convolve2d(img,kernel_jx,boundary='fill',mode='same')
	jy = signal.convolve2d(img,kernel_jy,boundary='fill',mode='same')
	return jx,jy

def compute_c(jx,jy):
	c = [[0,0],[0,0]]
	c[0][0] = np.sum(np.square(jx),dtype=np.int32)
	c[0][1] = c[1][0] = np.sum(np.multiply(jx,jy),dtype=np.int32)
	c[1][1] = np.sum(np.square(jy),dtype=np.int32)
	return c

def filter_points(points,threshold,offset):
	new_points = []
	points = sorted(points,key = lambda x : x[2],reverse=True)
	maxVal = points[0][2]
	threshold = maxVal * threshold
	points = [x for x in points if x[2] > threshold]
	
	while points:
		new_points.append(points[0])
		points = [point for point in points[1:] if point[0] > new_points[-1][0] + offset or point[1] > new_points[-1][1] + offset or point[0] < new_points[-1][0] - offset or point[1] < new_points[-1][1] - offset]
		#print(len(points))
	return new_points

def display_corners(img,corners):
	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')

	# Show the image
	ax.imshow(img)

	# Now, loop through coord arrays, and create a circle at each x,y pair
	for xx,yy,zz in corners:
		circ = Circle((xx,yy),5,fill=False)
		ax.add_artist(circ)

	# Show the image	
	plt.show()
		

def detect_corners(img,sigma,threshold,n):
	possible_corners = []
	rows = columns = 2*n +1
	offset=rows//2
	smooth_img = gaussian_filter(img,sigma)
	jx,jy = np.gradient(smooth_img)
	ex = jx**2
	ey = jy**2
	exey = jx*jy
	for i in range(offset,len(img)-offset):
		for j in range(offset,len(img)-offset):
			#neighbourhood = smooth_img[i-offset:i+offset+1,j-offset:j+offset+1]
			# jx,jy = gradients(neighbourhood)
			# c_matrix = compute_c(jx,jy)
			# eigenVals = np.linalg.eig(c_matrix)
			#print(eigenVals)
			ex_window = ex[i-offset:i+offset+1,j-offset:j+offset+1]
			ey_window = ey[i-offset:i+offset+1,j-offset:j+offset+1]
			exey_window = exey[i-offset:i+offset+1,j-offset:j+offset+1]
			sum_ex = ex_window.sum()
			sum_ey = ey_window.sum()
			sum_exey = exey_window.sum()
			c_matrix = [[sum_ex,sum_exey],[sum_exey,sum_ey]]
			eigenVals = np.linalg.eig(c_matrix)
			#print(eigenVals)
			del2 = min(eigenVals[0])
			possible_corners.append([i,j,del2])
	corners = filter_points(possible_corners,threshold,offset)
	display_corners(smooth_img,corners)
	print(corners)
	

if __name__ == "__main__":
	img = cv2.imread("Building1.jpg",0)
	detect_corners(img,3,0.2,2)