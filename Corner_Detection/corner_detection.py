import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from math import sqrt,pi,degrees,atan
import cv2


def compute_c(ex,ey,exey,i,j,offset):
	ex_window = ex[i-offset:i+offset+1,j-offset:j+offset+1]
	ey_window = ey[i-offset:i+offset+1,j-offset:j+offset+1]
	exey_window = exey[i-offset:i+offset+1,j-offset:j+offset+1]
	sum_ex = ex_window.sum()
	sum_ey = ey_window.sum()
	sum_exey = exey_window.sum()
	c_matrix = [[sum_ex,sum_exey],[sum_exey,sum_ey]]
	return c_matrix

def filter_points(points,threshold,offset):
	new_points = []
	points = sorted(points,key = lambda x : x[2],reverse=True)
	maxVal = points[0][2]
	threshold = maxVal * threshold
	points = [x for x in points if x[2] > threshold]
	
	while points:
		new_points.append(points[0])
		points = [point for point in points[1:] if point[0] > new_points[-1][0] + offset or point[1] > new_points[-1][1] + offset or point[0] < new_points[-1][0] - offset or point[1] < new_points[-1][1] - offset]
		
	return new_points

def draw_rectangle(img,x,y):
	for i in range(-2,3):
		img[x+i][y-2] = 255
		img[x+i][y+2] = 255
	
	for j in range(-2,3):
		img[x-2][y+j] = 255
		img[x+2][y+j] = 255
	

def display_corners(img,corners):
	for x,y,z in corners:
		draw_rectangle(img,x,y)
	
	plt.gray()
	plt.imshow(img)
	
	plt.show()
		

def detect_corners(img,sigma,threshold,n):
	possible_corners = []
	size = 2*n +1
	offset=size//2

	smooth_img = ndimage.gaussian_filter(img,sigma)
	jx,jy = np.gradient(smooth_img)
	ex = jx**2
	ey = jy**2
	exey = jx*jy
	for i in range(offset,len(img)-offset):
		for j in range(offset,len(img)-offset):
			c_matrix = compute_c(ex,ey,exey,i,j,offset)
			eigenVals = np.linalg.eig(c_matrix)
			del2 = min(eigenVals[0])
			possible_corners.append([i,j,del2])
	corners = filter_points(possible_corners,threshold,size)
	display_corners(smooth_img,corners)
	#print(corners)
	

if __name__ == "__main__":
	img = cv2.imread("CheckerBoard.jpg",0)
	detect_corners(img,1,0.1,4)
	img1 = cv2.imread("Building1.jpg",0)
	detect_corners(img1,1,0.1,4)