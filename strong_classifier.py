import numpy as np
import cv2
import math
import os

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
	feature_info = np.load("./dataset/feature_location_type.npy")
	weak_classifiers = np.load("./dataset/weak_classifiers_2.npy")
	#print(feature_info[0:2])
	#print(weak_classifiers[0:2])
	#print(len(weak_classifiers))
	#test_img1 = read_img("./test/face/cmu_0192.pgm")
	#test_img2 = read_img("./test/non-face/cmu_4637.pgm")
	#I_img1 = Integral_image(test_img1)
	#I_img2 = Integral_image(test_img2)

	positive_test_path = "./test/face/"
	negative_test_path = "./test/non-face"
	positive_files = os.listdir(negative_test_path)
	#alphas = []
	#classifier_results = []
	total_count  = len(positive_files)
	accuracy = []
	positive_files.remove('.DS_Store')

	for file in positive_files:
		alphas = []
		classifier_results = []
		test_img = read_img("./test/non-face/"+file)
		print(file)
		I_img = Integral_image(test_img)
		for classifier in weak_classifiers:
			feature_num = int(classifier[2])
			feature_type_info = feature_info[feature_num]
			#print(feature_num)
			#print(feature_type_info)
			i = feature_type_info[0]
			j = feature_type_info[1]
			h = feature_type_info[2]
			w = feature_type_info[3]
			t = feature_type_info[4]
			feature = get_feature(i,j,h,w,t,I_img)
			#print(feature)
			alpha_t = math.log(1/classifier[0])
			alphas.append(alpha_t)
			#print("Polarity: ")
			#print(classifier[3])
			#print("feature: ")
			#print(feature)
			#print("threshold: ")
			#print(classifier[1])
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
			#print(classifier[1])
			#print(classifier[3])
		#print(classifier_results)
		#print(alphas)
		classifier_results = np.array(classifier_results)
		alphas = np.array(alphas)

		alpha_sum = (1/2)*np.sum(alphas)
		strong_classifier = np.sum(np.multiply(alphas,classifier_results))
		print(strong_classifier)
		print(alpha_sum)
		if(abs(strong_classifier-alpha_sum)<2.0):
			accuracy.append(1)
		else:
			accuracy.append(0)
	accuracy = np.array(accuracy)
	print("total positves: ")
	print(np.sum(accuracy))
	print("total count of positves: ")
	print(total_count)


	#raise NotImplementedError
		



if __name__ == "__main__":
    main()