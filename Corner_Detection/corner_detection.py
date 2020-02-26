import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from math import sqrt,pi,degrees,atan
import cv2

#Computes the C matrix - [[Eex,Eexy],[Eexy],[Eey]]
#Takes in the gradient matrices ex (x gradient squared), ey (y gradient squared), exey (product of x and y gradients),
#								location to build neighborhood, size of neighborhood and returns the c matrix

def compute_c(ex,ey,exey,i,j,offset):
	ex_window = ex[i-offset:i+offset+1,j-offset:j+offset+1]
	ey_window = ey[i-offset:i+offset+1,j-offset:j+offset+1]
	exey_window = exey[i-offset:i+offset+1,j-offset:j+offset+1]
	sum_ex = ex_window.sum()
	sum_ey = ey_window.sum()
	sum_exey = exey_window.sum()
	c_matrix = [[sum_ex,sum_exey],[sum_exey,sum_ey]]
	return c_matrix

#Filters corners based on threshold from possible corner points
#Threshold is calculated based on a percentage of the maximum delta value in the list
#Takes in the possible corner points, threshold ratio and offset of neighborhood
def filter_points(points,threshold,offset):
	new_points = []
	#Sort in decreasing order of eigenvalue delta
	points = sorted(points,key = lambda x : x[2],reverse=True)
	maxVal = points[0][2]
	#Compute threshold eigenvalue
	threshold = maxVal * threshold
	points = [x for x in points if x[2] > threshold]
	
	#Remove all lower eigenvalue points in the neighbourhood
	while points:
		new_points.append(points[0])
		points = [point for point in points[1:] if point[0] > new_points[-1][0] + offset or point[1] > new_points[-1][1] + offset or point[0] < new_points[-1][0] - offset or point[1] < new_points[-1][1] - offset]
		
	return new_points

#Constructs a rectangle around the corner point. Size of the rectangle is the size of the neighbourhood
#Takes in the image, corner point co-ordinates and the offset for neighbourhood
def draw_rectangle(img,x,y,offset):
	for i in range(-offset,offset+1):
		img[x+i][y-offset] = 255
		img[x+i][y+offset] = 255
	
	for j in range(-offset,offset+1):
		img[x-offset][y+j] = 255
		img[x+offset][y+j] = 255
	

#Represents corners on the image and displays it on a plot
#Takes in the image, corner points and offset of neighbourhood
def display_corners(img,corners,offset):
	for x,y,z in corners:
		draw_rectangle(img,x,y,offset)
	
	plt.gray()
	plt.imshow(img)
	
	plt.show()
		
#Performs corner detection
#Takes in the input image, standard deviation value for gaussian smoothing, threshold value for corners and size of neighbourhood
def detect_corners(img,sigma,threshold,n):
	possible_corners = []
	#Determine number of pixels and offset for neighbourhood
	size = 2*n +1
	offset=size//2
	
	#Gaussian smoothing
	smooth_img = ndimage.gaussian_filter(img,sigma)

	#Obtain values for c matrix
	jx,jy = np.gradient(smooth_img)
	ex = jx**2
	ey = jy**2
	exey = jx*jy

	#Detect corners for each point
	for i in range(offset,len(img)-offset):
		for j in range(offset,len(img[0])-offset):
			c_matrix = compute_c(ex,ey,exey,i,j,offset)
			eigenVals = np.linalg.eig(c_matrix)
			#Smaller eigenvalue delta 2
			del2 = min(eigenVals[0])
			possible_corners.append([i,j,del2])
	corners = filter_points(possible_corners,threshold,size)
	display_corners(smooth_img,corners,offset)
	#print(corners)
	
#Test cases
if __name__ == "__main__":
	img = cv2.imread("CheckerBoard.jpg",0)
	detect_corners(img,1,0.1,4)
	img1 = cv2.imread("Building1.jpg",0)
	detect_corners(img1,1,0.1,4)