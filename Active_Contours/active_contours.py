import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from operator import itemgetter
from math import hypot,fabs,inf
import os
import sys
import cv2

"""
Global variables
points - Stores all contour points
corners - Stores corner points of contour
avg_dist - Average distance between contour points
"""
points = []
corners = []
avg_dist = 0

"""
Performs a gaussian smoothing and returns the image strength/magnitude
Parameters - Input image, sigma for gaussian smoothing
"""
def img_strength(img,sigma):
	res = ndimage.gaussian_filter(img.astype(float),3)
	jx,jy = np.gradient(res)
	
	
	img_str = np.sqrt(np.square(jx) + np.square(jy))
	return img_str

"""
Handles click event on the image and records points clicked
"""
def onclick(event):
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    plt.scatter([ix], [iy], s = 2, c='r')
    plt.draw()
    global points
    points.append((iy, ix))

"""
Obtains initial points for contour
Parameters - Input image on which contour will be placed
"""
def get_points(image):
	fig = plt.figure('ACTIVE CONTOURS')
	plt.gray()
	plt.axis("off")
	plt.imshow(image)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	fig.canvas.mpl_disconnect(cid)

"""
Fills the gap between input contour points by placing points 5px distance apart along the line
"""
def interpolate():
	temp_points = []
	global points
	num_points = len(points)
	for ind,point in enumerate(points):
		temp_points.append(point)
		nxt_point = points[(ind+1)%num_points]
		distance = hypot(nxt_point[0]-point[0],nxt_point[1]-point[1])
		if distance > 5:
			gaps = int(distance/5)
			gapx = (nxt_point[0]-point[0])//(gaps+1)
			gapy = (nxt_point[1]-point[1])//(gaps+1)
			for i in range(gaps):
				x = point[0] + gapx*i
				y = point[1] + gapy*i
				temp_points.append((x,y))
	points = temp_points


"""
Displays the input image with contour points overlayed on it. Contour points are red and corner points are blue
Parameters - Input image on which contour is placed
"""
def display_contour(img):
	global points
	plt.gray()
	plt.axis("off")
	plt.imshow(img)
	for point in points:
		plt.scatter(point[1],point[0],s=2,c='r')
	
	for point in corners:
		plt.scatter(point[1],point[0],s=2,c='b')
	plt.show()


"""
Computes average distance between contour points
"""
def average_dist():
	d = 0
	global points
	global avg_dist
	num_points = len(points)
	for ind,point in enumerate(points):
		nxt_point = points[(ind+1)%num_points]
		d += hypot(nxt_point[0]-point[0],nxt_point[1]-point[1])
	avg_dist = d/num_points

"""
Obtains neighbouring points around a pixel/point
Parameters - center point of neighbourhood, neighbourhood size
"""
def get_neighbours(index,neigh_size):
	global points
	offset = neigh_size//2
	center = points[index]
	neighbours = [(center[0]+x,center[1]+y) for x in range(-offset,offset+1) for y in range(-offset,offset+1)]
	
	return neighbours

"""
Calculates the continuity term between neighbouring points and previous and next contour points
Parameters - index of the current contour point, neighbours of the current contour point
"""
def continuity_term(index,neighbours):
	global points
	global avg_dist
	
	num_points = len(points)
	prev = points[(index-1)%num_points]
	neighbours_continuity = [fabs(avg_dist-hypot(nei[0]-prev[0],nei[1]-prev[1])) for nei in neighbours]
	
	return neighbours_continuity

"""
Calculates the curvature term between neighbouring points and previous and next contour points
Parameters - index of the current contour point, neighbours of the current contour point
"""
def curvature_term(index,neighbours):
	global points
	num_points = len(points)
	prev = points[(index-1)%num_points]
	nxt = points[(index+1)%num_points]
	neighbours_curvature = []
	for nei in neighbours:
		x_term = prev[0] + nxt[0] - nei[0]*2
		y_term = prev[1] + nxt[1] - nei[1]*2
		curvature = x_term**2 + y_term**2
		neighbours_curvature.append(curvature)
	
	return neighbours_curvature


"""
Calculates the strength term of the neighbouring points
Parameters - strength/magnitude matrix, neighbours of the current contour point
"""
def image_term(strength,neighbours):
	
	neighbours_strength = list(strength[nei[0],nei[1]] for nei in neighbours)
	
	maxVal = max(neighbours_strength)
	minVal = min(neighbours_strength)
	if(maxVal - minVal) < 5:
		minVal = maxVal - 5
	neighbours_image = list((minVal - strength[nei[0],nei[1]])/(maxVal-minVal) for nei in neighbours)
	
	return neighbours_image

"""
Function to detect maximum curvature between 3 points in the contour thereby detecting corners
Parameters - Index of the current contour point
"""
def maximum_curvature(ind):
	global points
	num_points = len(points)
	prev = points[(ind-1)%num_points]
	cur = points[ind]
	nxt = points[(ind+1)%num_points]

	u1 = (cur[0]-prev[0],cur[1]-prev[1])
	u2 = (nxt[0]-cur[0],nxt[1]-cur[1])

	u1_dist = hypot(u1[0]-0,u1[1]-0)
	u2_dist = hypot(u2[0]-0,u2[1]-0)

	u1_norm = (u1[0]/u1_dist if u1_dist >0 else 0,u1[1]/u1_dist if u1_dist >0 else 0)
	u2_norm = (u2[0]/u2_dist if u2_dist >0 else 0,u2[1]/u2_dist if u2_dist > 0 else 0)

	return (u1_norm[0]-u2_norm[0])**2 + (u1_norm[1]-u2_norm[1])**2	


"""
Greedy algorithm to detect active contours
Parameters - Input image,strength matrix, alpha value for continuity, beta value for curvature, gamma value for strength,
			 size of neighbourhood, threshold for detecting corner curvature, strength threshold for corner, points threshold
			 for minimum number of contour points to move for iterating
"""
def greedy_contours(img,strength,alpha_val,beta_val,gamma_val,neigh_size,curv_threshold,strength_threshold,pts_threshold):
	global points
	
	num_points = len(points)
	alpha = [alpha_val]*num_points
	beta = [beta_val]*num_points
	gamma = [gamma_val]*num_points
	curvature = [0]*num_points
	count = 0
	prev_ptsmoved = 0
	repeat = 0
	while True:
		ptsmoved = 0
		for i in range(num_points):
			average_dist()
			min_energy = inf
			neighbours = get_neighbours(i,neigh_size)
			neighbours_continuity = continuity_term(i,neighbours)
			neighbours_curvature = curvature_term(i,neighbours)
			neighbours_image = image_term(strength,neighbours)
			cur_point = points[i]
			cur_energy = 0
			
			for j in range(len(neighbours)):
				energy_j = alpha[i] * neighbours_continuity[j]/max(neighbours_continuity) + beta[i] * neighbours_curvature[j]/max(neighbours_curvature) + gamma[i] * neighbours_image[j]
				if neighbours[j] == cur_point:
					cur_energy = energy_j
				if energy_j < min_energy:
					min_point = neighbours[j]
					min_energy = energy_j
			
			
			if min_point != cur_point and min_energy != cur_energy:
				points[i] = min_point
				ptsmoved += 1
		
		count += 1
		#Display contours only once in 20 iterations
		if count %20 == 0:
			display_contour(img)

		for i in range(num_points):
			curvature[i] = maximum_curvature(i)
		
		for i in range(num_points):
			prev = curvature[(i-1)%num_points]
			cur = curvature[i]
			nxt = curvature[(i+1)%num_points]
			if cur > prev and cur > nxt and cur > curv_threshold and strength[points[i][0],points[i][1]] > strength_threshold:
				beta[i] = 0
				corners.append(points[i])
		
		#If number of points moved doesn't change, we need to stop iterating
		if prev_ptsmoved == ptsmoved:
			repeat += 1
		
		prev_ptsmoved = ptsmoved
		if ptsmoved < pts_threshold * num_points or repeat >= 40:
			break


"""
Detects if input is file or folder and runs the greedy_contours function on it
If input is a folder, the algorithm is run on all the files inside the folder assuming the files are images
Parameters - Input image,sigma for gaussian smoothing, alpha value for continuity, beta value for curvature, gamma value for strength,
			 size of neighbourhood, threshold for detecting corner curvature, strength threshold for corner, points threshold
			 for minimum number of contour points to move for iterating
"""
def active_contours(img,sigma=3,alpha_val=1,beta_val=1,gamma_val=1,neigh_size=9,curv_threshold=0.3,strength_threshold=10,pts_threshold=0.1):
	filepath = sys.path[0] + "\\" + img
	if os.path.isdir(filepath):
		files = os.listdir(filepath)
		images = [(filepath + "\\" + temp) for temp in files if temp[-3:] == "jpg" or temp[-3:] == "png"]
		print(images)
		for i in range(len(images)):
			global corners
			corners = []
			image = cv2.imread(images[i],0)
			if i == 0:
				get_points(image)
				interpolate()
			strength = img_strength(image,sigma)
			greedy_contours(image,strength,alpha_val,beta_val,gamma_val,neigh_size,curv_threshold,strength_threshold,pts_threshold)
			print("Completed " + images[i])
			display_contour(image)
	else:
		image = cv2.imread(filepath,0)
		strength = img_strength(image,sigma)

		get_points(image)
		interpolate()

		greedy_contours(image,strength,alpha_val,beta_val,gamma_val,neigh_size,curv_threshold,strength_threshold,pts_threshold)
		print("Completed "+ img)
		display_contour(image)


#TEST CASES
if __name__ == "__main__":
	active_contours("Images1through8/image1.jpg")
