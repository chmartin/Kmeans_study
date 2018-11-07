# Load Libraries
import numpy as np

# Function to find centroids

def findClosestCentroids (X, centroids):
	# number of centroids
	K = centroids.shape[0]
	#number of datapoints
	m = X.shape[0]
	#indicies to be returned
	indicies = np.zeros(X.shape[0],dtype=np.int32)
	
	# Loop over examples to find closest centroid
	for i in xrange(0,m):
		dist = np.sqrt(np.sum(np.square(centroids - X[i]),axis=1))
		indicies[i] = np.argmin(dist)
		
	return indicies