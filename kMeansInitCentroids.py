# Load Libraries
import numpy as np
import random

# Function to find centroids

def kMeansInitCentroids(X, num_centroids):
	#number of datapoints
	m = X.shape[0]
	# Centroids to be returned
	centroids = np.zeros((num_centroids, X.shape[1]),dtype=np.float32)
	# pick K random pixels to be starting centroid colors
	centroids = np.array(random.sample(X,num_centroids))
	return centroids