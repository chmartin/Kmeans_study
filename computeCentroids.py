# Load Libraries
import numpy as np

# Function to find centroids

def computeCentroids (X, indicies, num_centroids):
	#number of datapoints
	m = X.shape[0]
	#number of dimensions
	n = X.shape[1]
	#centroids to return
	centroids = np.zeros((num_centroids, n),dtype=np.float32)
	
	#Loop over centroids to compute its mean
	for i in xrange(0,num_centroids):
		index_array = indicies==i
		X_i = X[index_array]
		centroids[i] = X_i.mean(axis=0)
	return centroids