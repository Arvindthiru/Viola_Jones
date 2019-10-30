import glob
import math
import cv2
import numpy as np

def make_positive(x):
	if(x<0):
		return 0
	else:
		return x

files = glob.glob("./FDDB-folds/*ellipseList.txt")
c = 0
for file in files:
	#print(file)
	f = open(file,'r')
	lines = f.readlines()
	i = 0
	while(i<len(lines)):
		#print(lines[i].strip("\n"))
		if(lines[i].find("/")>0):
			count = int(lines[i+1].strip("\n"))
			#print(count)
			j = i+2
			end = j + count
			while(j < end):
				img_p = "./originalPics/"+lines[i].strip("\n")+".jpg"
				img = cv2.imread(img_p)
				s = np.shape(img)
				h = s[0]
				w = s[1]
				#raise NotImplementedError
				el_string = lines[j].strip("\n")
				el_data = el_string.split(" ")
				#print(el_data)
				major = float(el_data[0])
				minor = float(el_data[1])
				angle = float(el_data[2])
				x = float(el_data[3])
				y = float(el_data[4])
				cosine = math.cos(math.radians(abs(angle)))
				height = 2*major*cosine
				breadth = 2*minor*cosine
				#print(major,minor,angle,x,y)
				lcx = int(max(0,x-(breadth/2)))
				lcy = int(max(0,y-(height/2)))
				rcx = int(min(w-1,x+(breadth/2)))
				rcy = int(min(h-1,y+(height/2)))
				
				m=abs(abs(rcy-lcy)-abs(rcx-lcx))

				#y2 rcy y1 lcy x2 rcx x1 lcx
				#margin=abs(abs(y2-y1)-abs(x2-x1))
				r_x, r_y = 0.1, 0.4

				if(abs(rcy-lcy)>abs(rcx-lcx)):
				# y is greater
					rcy = math.floor(make_positive(rcy-m*r_y*0.25*2))
					lcy = math.floor(make_positive(lcy+m*r_y*0.75*2))
					rcx = math.floor(make_positive(rcx+m*r_x))
					lcx = math.floor(make_positive(lcx-m*r_x))
				else:
				# x is greater
					rcy = math.floor(make_positive(rcy+m*r_y))
					lcy = math.floor(make_positive(lcy-m*r_y))
					rcx = math.floor(make_positive(rcx-m*r_x))
					lcx = math.floor(make_positive(lcx+m*r_x))

				img_path = "./originalPics/"+lines[i].strip("\n")+".jpg\n"
				values = str(lcx)+" "+str(lcy)+" "+str(rcx)+" "+str(rcy)+"\n"
				#print(values)
				#print(img_path)
				new_file = open("./Square_annotations.txt","a+")
				new_file.write(img_path)
				new_file.write(values)
				c = c+1
				#img = cv2.imread(img_path)
				#cv2.rectangle(img, (int(lcx), int(lcy)), (int(rcx), int(rcy)), (0,0,255), 3)
				#cv2.imshow('image',img)
				#cv2.waitKey(0)
				#raise NotImplementedError
				#print(major,minor,angle,x,y)
				j = j+1
			i = end
print(c)