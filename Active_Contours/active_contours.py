import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,signal
from operator import itemgetter
from math import hypot,fabs,inf
import cv2

points = []
avg_dist = 0

def img_strength(img):

	jx,jy = np.gradient(img)
	img_str = np.hypot(jx,jy)
	return img_str


def onclick(event):
    global ix, iy
    ix, iy = int(event.xdata), int(event.ydata)
    # print(ix, iy)
    plt.scatter([ix], [iy], s = 2, c='r')
    plt.draw()
    global points
    points.append((iy, ix))

def get_points(image):
	fig = plt.figure('ACTIVE CONTOURS')
	plt.gray()
	plt.imshow(image)
	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plt.show()
	fig.canvas.mpl_disconnect(cid)

def interpolate():
	temp_points = []
	global points
	num_points = len(points)
	for ind,point in enumerate(points):
		temp_points.append(point)
		nxt_point = points[(ind+1)%num_points]
		distance = int(hypot(nxt_point[0]-point[0],nxt_point[1]-point[1]))
		if distance > 5:
			gaps = distance//5
			gapx = (nxt_point[0]-point[0])//(gaps+1)
			gapy = (nxt_point[1]-point[1])//(gaps+1)
			for i in range(1,gaps+1):
				x = point[0] + gapx*i
				y = point[1] + gapy*i
				temp_points.append((x,y))
	points = temp_points



def display_contour(img):
	global points
	plt.gray()
	plt.imshow(img)
	for point in points:
		plt.scatter(point[1],point[0],s=2,c='r')
	plt.show()


def average_dist():
	d = 0
	global points
	global avg_dist
	num_points = len(points)
	for ind,point in enumerate(points):
		nxt_point = points[(ind+1)%num_points]
		d += hypot(nxt_point[0]-point[0],nxt_point[1]-point[1])
	avg_dist = d/num_points

def get_neighbours(index):
	global points
	center = points[index]
	neighbours = [(center[0]+x,center[1]+y) for x in range(-1,2) for y in range(-1,2)]
	print(index)
	print(neighbours)
	return neighbours

def continuity_term(index,neighbours):
	global points
	global avg_dist
	num_points = len(points)
	prev = points[(index-1)%num_points]
	neighbours_continuity = [fabs(avg_dist-hypot(nei[0]-prev[0],nei[1]-prev[1])) for nei in neighbours]
	return neighbours_continuity

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

def image_term(strength,neighbours):
	neighbours_strength = [strength[nei[0],nei[1]] for nei in neighbours]
	maxVal = max(neighbours_strength)
	minVal = min(neighbours_strength)
	if(maxVal - minVal) < 5:
		minVal = maxVal - 5
	neighbours_image = [(minVal - strength[nei[0],nei[1]])/(maxVal-minVal) for nei in neighbours]
	return neighbours_image

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


def greedy_contours(img,strength):
	global points
	alpha = gamma = 1
	num_points = len(points)
	beta = [1]*num_points
	curvature = [0]*num_points
	count = 0
	while True:
		ptsmoved = 0
		print(points)
		for i in range(num_points):
			average_dist()
			min_energy = inf
			neighbours = get_neighbours(i)
			neighbours_continuity = continuity_term(i,neighbours)
			neighbours_curvature = curvature_term(i,neighbours)
			neighbours_image = image_term(strength,neighbours)
			cur_point = points[i]
			cur_energy = 0
			min_point = cur_point
			for j in range(len(neighbours)):
				energy_j = alpha * neighbours_continuity[j]/max(neighbours_continuity) + beta[i] * neighbours_curvature[j]/max(neighbours_curvature) + gamma * neighbours_image[j]/max(neighbours_image)
				if neighbours[j] == cur_point:
					cur_energy = energy_j
				if(energy_j < min_energy):
					min_point = neighbours[j]
					min_energy = energy_j
			
			if min_point != cur_point and min_energy < cur_energy:
				points[i] = min_point
				ptsmoved += 1
		print(ptsmoved)
		count += 1
		if count %5 == 0:
			display_contour(img)

		for i in range(num_points):
			curvature[i] = maximum_curvature(i)
		
		for i in range(num_points):
			prev = curvature[(i-1)%num_points]
			cur = curvature[i]
			nxt = curvature[(i+1)%num_points]
			if cur > prev and cur > nxt and cur > 0.3 and strength[points[i][0],points[i][1]] > 10:
				beta[i] = 0

		# if ptsmoved < 0.1 * num_points:
		# 	break
		


if __name__ == "__main__":
	img = cv2.imread("Images1through8/image1.jpg",0)
	img = ndimage.gaussian_filter(img,1)
	strength = img_strength(img)

	get_points(img)
	interpolate()
	greedy_contours(img,strength)
	
