import numpy as np
import cv2
import math
import os
import json
import sys

def n_m_s(boxes,overlapThresh):
	if len(boxes) == 0:
		return []
	pick = []
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		for pos in range(0, last):
			j = idxs[pos]
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			overlap = float(w * h) / area[j]
			if overlap > overlapThresh:
				suppress.append(pos)
		idxs = np.delete(idxs, suppress)
	return boxes[pick]

def classify_img(test_img,feature_info,weak_classifiers):
	alphas = []
	classifier_results = []
	I_img = Integral_image(test_img)
	for classifier in weak_classifiers:
		feature_num = int(classifier[2])
		feature_type_info = feature_info[feature_num]
		i = feature_type_info[0]
		j = feature_type_info[1]
		h = feature_type_info[2]
		w = feature_type_info[3]
		t = feature_type_info[4]
		feature = get_feature(i,j,h,w,t,I_img)
		alpha_t = math.log(1.0/classifier[0])
		alphas.append(alpha_t)
		if( classifier[3] == 0):
			if(feature > classifier[1]):
				classifier_results.append(1)
			else:
				classifier_results.append(0)
		elif( classifier[3] == 1):
			if(feature < classifier[1]):
				classifier_results.append(1)
			else:
				classifier_results.append(0)
	classifier_results = np.array(classifier_results)
	alphas = np.array(alphas)

	alpha_sum = (1/2)*np.sum(alphas)
	strong_classifier = np.sum(np.multiply(alphas,classifier_results))
	alpha_sum = (1/2)*np.sum(alphas)
	strong_classifier = np.sum(np.multiply(alphas,classifier_results))
	if(abs(strong_classifier-alpha_sum)>0):
		return 1
	else:
		return 0

def get_feature(i,j,h,w,t,Iimg1):
	if(t == 1):
		c = Rect_value(i,j,h,w,Iimg1)
		r = Rect_value(i,j+w,h,w,Iimg1)
		feature = abs(c-r)
	if(t == 2):
		c = Rect_value(i,j,h,w,Iimg1)
		d = Rect_value(i+h,j,h,w,Iimg1)
		feature = abs(c-d)
	if(t == 3):
		c = Rect_value(i,j,h,w,Iimg1)
		r = Rect_value(i,j+w,h,w,Iimg1)
		nr = Rect_value(i,j+2*w,h,w,Iimg1)
		feature = abs(c+nr-r)
	if(t == 4):
		c = Rect_value(i,j,h,w,Iimg1)
		d = Rect_value(i+h,j,h,w,Iimg1)
		nd = Rect_value(i+2*h,j,h,w,Iimg1)
		feature = abs(c+nd-d)
	if(t == 5):
		c = Rect_value(i,j,h,w,Iimg1)
		r = Rect_value(i,j+w,h,w,Iimg1)
		d = Rect_value(i+h,j,h,w,Iimg1)
		dr = Rect_value(i+h,j+w,h,w,Iimg1)
		feature = abs(c+dr-r-d)
	return feature

def read_img(img):
    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    return img

def Integral_image(img):
	I_img = []
	r_sum = 0
	row = []
	for rows in img:
		rows = np.array(rows)
		row = []
		sum = 0
		for elem in rows:
			sum = sum + elem
			row.append(sum)
		I_img.append(row)
	r = len(I_img)
	c = len(I_img[0])
	for i in range(1,r):
		for j in range(0,c):
			I_img[i][j] = I_img[i][j] + I_img[i-1][j]
	return I_img

def Rect_value(i,j,h,w,Int_img):
	Sum = 0
	if(i == 0 and j ==0):
		Sum =  Int_img[i+h-1][j+w-1]
	elif(i == 0 and j!=0):
		Sum = Int_img[i+h-1][j+w-1] - Int_img[i+h-1][j-1]
	elif(j == 0 and i!=0):
		Sum = Int_img[i+h-1][j+w-1] - Int_img[i-1][j+w-1]
	else:
		D = Int_img[i-1][j-1]
		C= Int_img[i-1][j+w-1]
		B = Int_img[i+h-1][j-1]
		A = Int_img[i+h-1][j+w-1]
		Sum = A - B - C + D
	return Sum

def main():
	feature_info = np.load("./FDDB-dataset/feature_location_type.npy")
	weak_classifiers = np.load("./FDDB-dataset/weak_classifiers.npy")
	file_directory = str(sys.argv[1])
	#raise NotImplementedError
	file_path = "./"+file_directory
	files = os.listdir(file_path)
	c = 1
	json_list = []
	for file in files:
		print("Finding faces for file " + file )
		final_test = read_img(file_path+"/"+file)
		shape = final_test.shape
		rows = shape[0]
		columns = shape[1]
		w_x = int(rows/4)
		w_y = int(rows/4)
		stride = int(rows/4)
		locations = []
		for w in range(w_x,rows,stride):
			for h in range(w_y,columns,stride):
				for i in range(0,rows-w_x,stride):
					for j in range(0,columns-w_y,stride):
						test_img = final_test[i:i+w,j:j+h]
						resized_img = cv2.resize(test_img,(19,19))
						result = classify_img(test_img,feature_info,weak_classifiers)
						if(result == 1):
							#print("found face")
							locations.append([i,j,i+w,j+h])
		locations = np.array(locations)
		best_loc = n_m_s(locations,0.2)
		print(len(best_loc))
		for i in best_loc:
			x = int(i[0])
			y = int(i[1])
			w = int(i[2])
			h = int(i[3])
			element = {"iname": file, "bbox":[x,y,w,h]} 
			json_list.append(element)
		#if(c == 2):
		#	break
		#c = c+1
	print(len(json_list))
	output_json = "results.json"
	json_list = list(json_list)
	with open(output_json,'w') as f:
		json.dump(json_list, f)

if __name__ == "__main__":
    main()