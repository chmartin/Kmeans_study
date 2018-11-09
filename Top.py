# Load libraries
import scipy.io as sio
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
import numpy as np
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from pympler import asizeof
from sklearn.cluster import KMeans
import time as time

# Step 1 show K-means in action
print 'Finding closest centroids.\n\n'

# load MATLAB file containing 2D data
mat_contents = sio.loadmat('ex7data2.mat')
X = mat_contents['X']

# Select an initial set of centroids and define the number of itterations
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3],[6, 2],[8, 5]],dtype=np.float32)
max_iters = 10

# Find the closest centroids for the examples using the initial_centroids
indicies = findClosestCentroids(X, initial_centroids)

print 'Closest centroids for the first 3 examples: \n'
print indicies[0], indicies[1], indicies[2]
print '\n(the closest centroids should be 0, 2, 1 respectively)\n'
raw_input("Press any key to continue")

print '\nComputing centroids means.\n\n'

#Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, indicies, K)

print 'Centroids computed after initial finding of closest centroids: \n'
print centroids[0],'\n', centroids[1],'\n', centroids[2]
print '\nthe centroids should be\n'
print '   [ 2.428301 3.157924 ]\n'
print '   [ 5.813503 2.633656 ]\n'
print '   [ 7.119387 3.616684 ]\n\n'
raw_input("Press any key to continue")

print '\nRunning K-Means clustering on example dataset.\n\n'

# Run K-Means algorithm. The 'true' at the end tells our function to plot the progress
centroids, indicies = runkMeans(X, initial_centroids, max_iters, True)

print '\nK-Means Done.\n\n'

# Step 2 Do K-Means on image colors for compression
print '\nRunning K-Means clustering on pixels from an image.\n\n'

# Open PNG image
A = mpimg.imread('bird_small.png')
A_size = asizeof.asizeof(A)

#Unroll 3D (rows x columns x [R,G,B]) into 2D ([R,G,B] x pixels)
A_flat = A.reshape(-1,3)

# Run your K-Means algorithm on this data
# Test values
K = 128; 
max_iters = 100;
#Set start time for runtime comparison
start_time = time.time()
# pick K random pixels to be starting centroid colors 
initial_centroids = kMeansInitCentroids(A_flat, K)

# Run K-Means algorithm.
centroids, indicies = runkMeans(A_flat, initial_centroids, max_iters, False)
Manual_time = time.time() - start_time

#Recover A
centroids_size = asizeof.asizeof(centroids)
indicies_size = asizeof.asizeof(indicies)
A_recovered = centroids[indicies]
A_recovered = A_recovered.reshape(A.shape[0],A.shape[1],3)

#Compute Size Reduction
after_size = centroids_size + indicies_size
reduction = float(after_size)/float(A_size)

#Plot to Compare
fig = pyplot.figure()
pyplot.clf()
pyplot.ion()
pyplot.axis("off")
pyplot.suptitle('Image Compression Comparison')
ax = pyplot.subplot(1,2,1)
ax.set_title('Before size: {0}'.format(A_size))
ax.axis("off")
ax.imshow(A)
ax = pyplot.subplot(1,2,2)
ax.set_title('My {0} colors, Size: {1}+{2}\n Frac Size {3:0.2f}'.format(K,centroids_size,indicies_size,reduction))
ax.axis("off")
pyplot.imshow(A_recovered)
raw_input("Press any key to continue")

#Compare to sklearn KMeans
start_time = time.time()
kmeans = KMeans(n_clusters=K,max_iter = max_iters,init='random').fit(A_flat)
Sklearn_time = time.time() - start_time
centroids_sklearn = kmeans.cluster_centers_
indicies_sklearn = kmeans.labels_
distortion = kmeans.inertia_

#print len(indicies_sklearn), len(indicies)
#print len(centroids_sklearn), len(centroids)
#print indicies_sklearn.dtype, indicies.dtype
#print centroids_sklearn.dtype, centroids.dtype

#Recover A
centroids_size_sklearn = asizeof.asizeof(centroids_sklearn)
indicies_size_sklearn = asizeof.asizeof(indicies_sklearn)
A_recovered_sklearn = centroids_sklearn[indicies_sklearn]
A_recovered_sklearn = A_recovered_sklearn.reshape(A.shape[0],A.shape[1],3)

#Compute Size Reduction
after_size_sklearn = centroids_size_sklearn + indicies_size_sklearn
reduction_sklearn = float(after_size_sklearn)/float(A_size)

#Plot to Compare
pyplot.clf()
pyplot.ion()
pyplot.axis("off")
pyplot.suptitle('Image Compression Comparison')
ax = pyplot.subplot(2,2,1)
ax.set_title('Before size: {0}'.format(A_size))
ax.axis("off")
ax.imshow(A)
ax = pyplot.subplot(2,2,2)
ax.set_title('My {0} colors, Size: {1}+{2}'.format(K,centroids_size,indicies_size))
ax.axis("off")
pyplot.imshow(A_recovered)
ax = pyplot.subplot(2,2,3)
ax.set_title('Sklearn {0} colors, Size: {1}+{2}'.format(K,centroids_size_sklearn,indicies_size_sklearn))
ax.axis("off")
pyplot.imshow(A_recovered_sklearn)
print 'Size comparison (as fraction of initial size)'
print 'Manual: {0:.2f}, Sklearn: {1:.2f}'.format(reduction,reduction_sklearn)
print 'Runtime comparison'
print 'Manual: {0:.4f}, Sklearn: {1:.4f}'.format(Manual_time,Sklearn_time)
raw_input("Press any key to continue")


#Determine best number of clusters
Kmax = 255
distortions = np.zeros(Kmax,dtype=np.int32)
Ks = list(xrange(1,Kmax+1,10))
for i in Ks:
	kmeans = KMeans(n_clusters=i).fit(A_flat)
	distortions[i-1] = kmeans.inertia_

pyplot.clf()
pyplot.ion()
pyplot.title('Clusters for image compression')
pyplot.plot(Ks,distortions)
pyplot.xlabel('Number of Colors')
pyplot.ylabel('Distortion')
pyplot.yscale('log')
pyplot.show()
raw_input("Press any key to continue")
