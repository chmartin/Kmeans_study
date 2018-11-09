# Load Libraries
import numpy as np
import matplotlib.pyplot as pyplot
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids

# Manual kMeans implementation. Plotting option for 2D data.

def runkMeans(X, initial_centroids, max_iters, plot_progress):
	""" My Manual kMeans implementation."""
	
	#number of datapoints
	m = X.shape[0]
	#number of dimensions
	n = X.shape[1]
	#number of centroids
	K = initial_centroids.shape[0]
	#indicies & centroids to be returned
	indicies = np.zeros(X.shape[0],dtype=np.int32)
	centroids = initial_centroids
	previous_centroids = centroids
	
	#setup figure for plotting
	if plot_progress == True:
		fig = pyplot.figure()
		pyplot.ion()
		ax = pyplot.axes()
		ax.title('K-means centroid evolution')
		ax.ylabel('Feature 2')
		ax.xlabel('Feature 1')
		pyplot.show()
	
	# Run K-Means, Loop over for number of iterations specified in input
	for i in xrange(0,max_iters):
		# Output progress
		print 'K-Means iteration ', i, '/', max_iters, ' ...\n'
		# For each example in X, assign it to the closest centroid
		indicies = findClosestCentroids(X, centroids)
		if plot_progress == True:
			ax.scatter(X[:,0],X[:,1],c=indicies, label='Data for centroid {0}'.format(indicies))
			ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='r', linewidth=2, label='Centroids')
			previous_centroids = centroids
			ax.legend()
			pyplot.draw()
			raw_input("Press any key to continue")
		# compute new centroids
		centroids = computeCentroids(X, indicies, K);
		
	return centroids, indicies
