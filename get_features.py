import argparse
import os
import cv2
import random
import numpy as np
import random

from random import shuffle

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

def Get_feature_location_type(img):

	shape = np.shape(img)
	#print(shape)
	rows = shape[0]
	columns = shape[1]
	features_l_t = []
	for i in range(0,rows):
		for j in range(0,columns):
			for w in range(1,columns+1):
				for h in range(1,rows+1):
					if(i+h-1<rows and j+2*w-1 < columns):
						features_l_t.append([i,j,h,w,1])
					if(i+2*h-1 < rows and j+w-1 <columns):
						features_l_t.append([i,j,h,w,2])
					if(i+h-1<rows and j+3*w-1 < columns):
						features_l_t.append([i,j,h,w,3])
					if(i+3*h-1 < rows and j+w-1< columns):
						features_l_t.append([i,j,h,w,4])
					if(j+2*w-1< columns and i+2*h-1 < rows):
						features_l_t.append([i,j,h,w,5])
	#print(len(features))
	return(features_l_t)

def Get_features(img):

	shape = np.shape(img)
	print("Shape of image: ")
	print(shape)
	Iimg = Integral_image(img)
	rows = shape[0]
	columns = shape[1]
	features = []
	features_l_t = []
	for i in range(0,rows):
		for j in range(0,columns):
			for w in range(1,columns+1):
				for h in range(1,rows+1):
					if(i+h-1<rows and j+2*w-1 < columns):
						c = Rect_value(i,j,h,w,Iimg)
						r = Rect_value(i,j+w,h,w,Iimg)
						#features.append([c,r])
						features.append(abs(c-r))
					if(i+2*h-1 < rows and j+w-1 <columns):
						c = Rect_value(i,j,h,w,Iimg)
						d = Rect_value(i+h,j,h,w,Iimg)
						#features.append([c,d])
						features.append(abs(c-d))
					if(i+h-1<rows and j+3*w-1 < columns):
						nr = Rect_value(i,j+2*w,h,w,Iimg)
						#features.append([c,r,nr])
						features.append(abs(c+nr-r))
					if(i+3*h-1 < rows and j+w-1< columns):
						nd = Rect_value(i+2*h,j,h,w,Iimg)
						#features.append([c,d,nd])
						features.append(abs(c+nd-d))
					if(j+2*w-1< columns and i+2*h-1 < rows):
						dr = Rect_value(i+h,j+w,h,w,Iimg)
						#features.append([c,r,d,dr])
						features.append(abs(c+dr-r-d))
	return(features)


def main():
	positive_file_path = "./FDDB_train/face/"
	negative_file_path = "./FDDB_train/non-face/"
	positive_files = os.listdir(positive_file_path)
	negative_files = os.listdir(negative_file_path)
	positive_files.remove('.DS_Store')
	pf = []
	nf = []
	for i in positive_files:
		pf.append([i,1])
	for i in negative_files:
		nf.append([i,0])
	m = len(pf)
	l = len(nf)
	print(m)
	print(l)
	pf.extend(nf)
	shuffle(pf)
	c = 1
	test_img = read_img("./FDDB_train/face/img1.jpg")
	feature_loc_type =  Get_feature_location_type(test_img)
	print(len(feature_loc_type))
	#raise NotImplementedError
	All_features = []
	for i in pf: 
		print("Iteration "+str(c))
		c = c+1
		print("Computing features for "+i[0])
		test_img = read_img("./FDDB_train/All/"+i[0])
		features = Get_features(test_img)
		print("Features length obtained : ")
		print(len(features))
		All_features.append(features)
	print("Length of All features matrix: ")
	print(len(All_features))
	np.save("./FDDB-dataset/features",All_features)
	np.save("./FDDB-dataset/target", pf)
	np.save("./FDDB-dataset/feature_location_type",feature_loc_type)


if __name__ == "__main__":
    main()