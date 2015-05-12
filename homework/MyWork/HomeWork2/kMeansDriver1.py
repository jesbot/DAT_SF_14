# K-Means Main driver.
#
# Purupose: Applies a basic K-means on the tulip data set.
#
# Author: Jose Solomon
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
	print 'iris data head:'
	# Make sure the data is correct
	print irisData.head()
	# Determine the number of features


	x = np.array([1, 2, 3, 4, 5])
	y = np.array([8, 8, 8, 8, 8])
	z = np.ones((9, 5))

	result = np.sqrt(((z-x)**2).sum(axis=0))

	print result


	return


def main():
	# Define a range of number of centroids
	kCenters = range(2,10,1)
	print kCenters[1]
	kMeans(irisDF, kCenters[1] )

if __name__ == '__main__':
	main()