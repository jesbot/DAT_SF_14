# K-Means Main driver.
#
# Purupose: Applies a basic K-means on the tulip data set.
#
# Author: Jose Solomon
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import datasets


# Load the data set into a dataframe
iris = datasets.load_iris()
# Now convert it to a pandas datafrem
irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)


# Main driver function
def kMeans(irisData, k):
	# print 'iris data head:'
	# Make sure the data is correct
	print irisData.head()
	
	# Determine the number of data points and features
	nPoints = irisData.shape[0]
	nFeatures = irisData.shape[1]

	# Create an initial number of random features
	initialCentroids= initializeCentroids(k, nPoints, nFeatures, irisData)
	print initialCentroids
	

	return

# Create an initial set of indices to get a starting set of 
# centroids
def initializeCentroids(kReq, kAvail, nFeatures, irisData):
	# Found a standard library to do what is needed
	indices = random.sample(range(kAvail), kReq)
	print indices

	# Convert 'Data Frame' to matrix
	irisMx = irisData.as_matrix()
	initialCentroids = np.zeros((kReq, nFeatures))

	# Store the correct centroids
	count = 0
	for index in indices:
		initialCentroids[count,:] = irisMx[index,:]
		count += 1

	return initialCentroids

def main():
	# Define a range of number of centroids
	kCenters = range(2,10,1)
	kMeans(irisDF, kCenters[2] )

if __name__ == '__main__':
	main()

# Reference commented code
	# x = np.array([1, 2, 3, 4, 5])
	# y = np.array([8, 8, 8, 8, 8])
	# z = np.ones((9, 5))

	# # Distance 
	# distance = np.sqrt(((z-x)**2).sum(axis=1))
	# print distance
	# single_point = [3, 4]
	# print single_point
	# points = np.arange(20).reshape((10,2))
	# print points
	# dist = (points - single_point)**2
	# dist = np.sum(dist, axis=1)
	# dist = np.sqrt(dist)

	# result = np.sqrt(((z-x)**2).sum(axis=0))

	# print result
