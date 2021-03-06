# Kmeans_study
This is a quick project studying Kmeans clustering using python.

The original idea was to recreate a Octave/Matlab Kmeans image compression exercise from the "Coursea Machine Learning Course by Andrew Ng" in Python. Then I wanted to do some quck comparison of my 'manual' Kmeans algorithm with the sklearn built in kmeans methods.

Running program 'Top.py' will do the following, usually pausing after each:

On 2D data from original exercise

1) Find nearest initial centroids for all datapoints
2) Compute mean of each centroid
3) Run manual Kmeans plotting centroid progression

On Imgae file from original exercise

1) Run custom random centroid initialization to pick random pixels as initial centroid colors
2) Run Kmeans to compress image
3) Plot before and after compression, comparing size difference
4) Compare original, manual and sklearn Kmeans implimentions as images and sizes
5) Run sklearn Kmeans with different numbers of centroids to plot distortion as a funciton of K
(NOTE: Part 5 can have long run time depending on range of centroids chosen.

How to run full exercize:

python Top.py

Note: The last step of the program take take a long time to run depending on your settings.

Python modules required:

scipy.io
matplotlib.pyplot
matplotlib.image
numpy
pympler
sklearn.cluster
time

Files:

Input data: ex7data.mat, bird_small.img (These are both from Coursera exercise)
Python files: Top.py, runkMeans.py, kMeansInitCentroids.py, findClosestCentroids.py, computeCentroids.py

Work In Progress:

Some runs will throw the following warning when running custom image compression:

/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/_methods.py:55: RuntimeWarning: Mean of empty slice.
  warnings.warn("Mean of empty slice.", RuntimeWarning)
/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/_methods.py:65: RuntimeWarning: invalid value encountered in true_divide
  ret, rcount, out=ret, casting='unsafe', subok=False)

This will lead to bad convergence.
