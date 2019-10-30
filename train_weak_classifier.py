import numpy as np
import cv2


def Get_normalized_weights(weights):
	sw = np.sum(weights)
	print(sw)
	return(weights/sw)

def Get_min(a,b):
	if(a<=b):
		return [a,0]
	else:
		return [b,1]

def Get_Classifier_error(norm_w,f,t,tp,tn):
	sp = 0.0
	sn = 0.0
	errors = []
	ft = []
	c = 0
	for i in range(0,len(f)):
		ft.append([f[i],norm_w[i],t[i]])
	ft = np.array(ft)
	ft = ft[ft[:,0].argsort()]
	minimum = 9999.0 
	polarity = 0
	for i in ft:
		if(i[2] == 1):
			sp = sp + i[1]
		if(i[2] == 0):
			sn = sn + i[1]
		left = sp + tn - sn
		right = sn + tp - sp
		e = Get_min(left,right)
		if(e[0] < minimum):
			minimum = e[0]
			threshold = i[0] 
			polarity = e[1]
	classifier = []
	if (polarity == 0):
		for i in f:
			if(i > threshold):
				classifier.append(1)
			else:
				classifier.append(0)
	else:
		for i in f:
			if(i < threshold):
				classifier.append(1)
			else:
				classifier.append(0)
	classifier = np.array(classifier)
	correctness = np.absolute(np.subtract(classifier,t))
	error = np.sum(np.multiply(norm_w,correctness))
	return error,threshold,polarity,correctness

def main():
	All_features = np.load("./FDDB-dataset/features.npy")
	target = np.load("./FDDB-dataset/target.npy")
	feature_info = np.load("./FDDB-dataset/feature_location_type.npy")
	print("Read required data from disk")
	weights = []
	positives = 0
	negatives = 0
	print(feature_info[0])
	for i in target:
		if(i[1] == '1'):
			positives = positives + 1
		if(i[1] == '0'):
			negatives = negatives + 1
	#raise NotImplementedError
	for i in target:
		if(i[1] == '1'):
			weights.append(1/(2*positives))
		if(i[1] == '0'):
			weights.append(1/(2*negatives))
	weights = np.array(weights)
	num_target = target[:,1]
	num_target = list(map(int,num_target))
	num_target = np.array(num_target)
	#raise NotImplementedError 
	weak_classifier_data = []
	for t in range(0,15):
		tpw = 0
		tnw = 0
		norm_weights = Get_normalized_weights(weights)
		for i in range(0,len(num_target)):
			if(num_target[i] == 1):
				tpw = tpw + norm_weights[i]
			if(num_target[i] == 0):
				tnw = tnw + norm_weights[i]
		min_error = 9999.0
		feature_threshold = 0
		feature_polarity = 0
		feature = 0
		for j in range(0,len(feature_info)):
			f = All_features[:,j]
			#raise NotImplementedError
			print("Iteration for feature: ",str(j))
			classifier_error, threshold, polarity, correctness = Get_Classifier_error(norm_weights,f,num_target,tpw,tnw)
			if(classifier_error < min_error):
				min_error = classifier_error
				feature_threshold = threshold
				feature_polarity = polarity
				feature = j
				best_correctness = correctness
		print("Minimum error for weak classifier: ")
		print(min_error)
		print("Classifier threshold: ")
		print(feature_threshold)
		print("Classifier Polarity: ")
		print(feature_polarity)
		print("feature number: ")
		print(feature)
		beta_t = min_error/(1-min_error)
		beta_array = []
		for i in range(0,len(best_correctness)):
			if(best_correctness[i] == 0):
				beta_array.append(beta_t)
			else:
				beta_array.append(1)
		beta_array = np.array(beta_array)
		print("Weights before updating: ")
		print(norm_weights[0:30])
		weights = np.multiply(norm_weights,beta_array)
		print("Weights after updating: ")
		print(weights[0:30])
		print("Classifier "+str(t)+"Done")
		weak_classifier_data.append([beta_t,feature_threshold,feature,feature_polarity])

	print(len(weak_classifier_data))
	np.save("./FDDB-dataset/weak_classifiers",weak_classifier_data)

if __name__ == "__main__":
    main()