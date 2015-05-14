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
MAXSEARCH = 100

# Load the data set into a dataframe
iris = datasets.load_iris()
# Now convert it to a pandas datafrem
irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
# Convert 'Data Frame' to matrix
irisMx = irisData.as_matrix()

# Main driver function
def kMeans(irisData, k):
	# Determine the number of features
	nFeatures = irisData.shape[1]

	# Let's use the data set to define min and max values
	minMax = np.zeros((2,nFeatures))
	minMax[0,:] = irisMx.min(0)
	minMax[1,:] = irisMx.max(0)
	# print minMax

	# Create an initial number of random features
	currentCentroids = initializeCentroids(k, minMax, nFeatures, irisMx)
	print currentCentroids

	# count iterations 
	nIterations = 0
	# set initial previous centroids
	previousCentroids = None

	# peform search
	while searchContinues(previousCentroids, currentCentroids, nIterations):
		# Update the old centroids with current centroids
		previousCentroids = currentCentroids
		# Update the iteration count
		nIterations += 1

		# Set each member of the data set to its respective cluster
		clusterLabels = findLabels(irisMx, currentCentroids)

		# Based on each of the cluster sets, find the centroids
		currentCentroids = getCentroids(irisMx, clusterLabels, k)

		# Print iterations
		print nIterations

	return currentCentroids

# Calculate new centroids
def getCentroids(irisMx, clusterLabel, k):
	# Initialize centroids
	centroids = np.zeros((k,irisMx.shape[1]))
	kRange = range(0,k)

	for kIndex in kRange:
		# Find those elements that belong to a given label
		indices = np.where(clusterLabel == kIndex)[0]
		# print indices
		# Take the geometric mean... per coordinate
		sumValues = irisMx[indices,:].sum(axis=0)
		# Note: if a cluster has no values associated with it
		# this will return Nan's
		centroids[kIndex,:] = sumValues/indices.size

	print centroids

	return centroids

# Convergence check function
def searchContinues(oldCentroids, currCentroids, iteration):
	keepSearching = True
	if iteration >= MAXSEARCH:
		keepSearching =  False  # stop the search : max iterations
	if np.array_equal(oldCentroids, currCentroids):
	    keepSearching = False  # stop search: centroids have not changed
	return keepSearching

# Find labels for each data entry
def findLabels(irisMX, currentCentroids):
	# distance array
	distanceMatrix = np.zeros((irisMX.shape[0],currentCentroids.shape[0]))
	# print distanceMatrix

	# Find euclidean distance for each entry to each centroid
	nCentroids = range(0, currentCentroids.shape[0])
	for label in nCentroids:
		distanceMatrix[:,label] = \
		  np.sqrt(((irisMX-currentCentroids[label,:])**2).sum(axis=1))

	# So now we have distances... find the minimum for each entry and that
	# is your label
	labels = np.argmin(distanceMatrix,axis=1)
	# print labels

	return labels


# Create an initial set of indices to get a starting set of 
# centroids
def initializeCentroids(kReq, minMax, nFeatures, irisMx):
	# Initialize centroids
	randomCentroids = np.zeros((kReq, nFeatures))
	kCentroids = range(0,kReq)

	# Now find random centroids
	for k in kCentroids:
		randomCentroids[k,:] = randomCentroid(minMax, nFeatures)


	allPresent = False
	while not allPresent:
		# check to see if all random centroids have at least one member
		labels = findLabels(irisMx, randomCentroids)
		print labels
		for k in kCentroids:
			if k not in labels:
				randomCentroids[k] = randomCentroid(minMax, nFeatures)
				break
			allPresent = True

	return randomCentroids

def randomCentroid(minMax, nFeatures):
	randomCentroid = np.zeros((1,nFeatures))
	nFeat = range(0,nFeatures)
	for feature in nFeat:
		randomCentroid[0, feature] = random.uniform(minMax[0,feature], \
			minMax[1,feature])
	return randomCentroid

# Copied from class
def plotCentroids(centroids):
	fig, axes = plt.subplots(nrows=2, ncols=3)

	colors = ['r','g','b']
	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
	    tmp.plot(x=0,y=1, kind='scatter', c=colors[i], ax=axes[0,0])

	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
	    tmp.plot(x=0,y=2, kind='scatter', c=colors[i], ax=axes[0,1])

	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
	    tmp.plot(x=0,y=3, kind='scatter', c=colors[i], ax=axes[0,2])
	    
	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
	    tmp.plot(x=1,y=2, kind='scatter', c=colors[i], ax=axes[1,0])

	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
	    tmp.plot(x=1,y=3, kind='scatter', c=colors[i], ax=axes[1,1])

	for i in range(3): 
	    tmp = iris_df[iris_df.Target == i]
		 tmp.plot(x=2,y=3, kind='scatter', c=colors[i], ax=axes[1,2])


def main():
	# Define a range of number of centroids
	kCenters = range(2,10,1)
	# Perform k-means
	centroids = kMeans(irisDF, kCenters[2])
	# Plot centroids
	plotCentroids(centroids)

if __name__ == '__main__':
	main
