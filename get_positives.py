import cv2

#import numpy as np
file = open("./Square_annotations.txt")
lines = file.readlines()
i = 0
k = 1
while(i<len(lines)):
	print(k)
	img_path = lines[i].strip("\n")
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	elem = lines[i+1].strip("\n")
	values = elem.split(" ")
	print(img_path)
	print(values)

	lcx = int(values[0])
	lcy = int(values[1])
	rcx = int(values[2])
	rcy = int(values[3])
	breadth = abs(rcx - lcx)
	length = abs(lcy - rcy)
	print(breadth,length)

	crop_img = img[lcy:lcy+length, lcx:lcx+breadth]
	img_name = "img"+str(k)+".jpg";
	resized_img = cv2.resize(crop_img,(19,19))
	k = k+1
	cv2.imwrite("./Resized_Positives/"+img_name,resized_img)

	i = i+2