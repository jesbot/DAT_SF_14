# processTrainingData.py
#
# Purpose: Goes through the lables for the training data and
# performs basic analysis. 
#  
# Author: Jose Solomon

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Load the csv data for all the training set
labelsData = pd.read_csv('trainLabels.csv')
labelsData = labelsData.iloc[:,0:2]
print labelsData.head(5)

# Load the csv data for the files that I do have
currentData = pd.read_csv('trainImages.csv')
currentData = currentData.iloc[:,0:1]
print currentData.head(5) 

# Create a data frame that consists of only the current data
# with the diagnosis from 'trainLables'
currentDF = labelsData.loc[labelsData['image'].isin(currentData.iloc[:,0])]
print currentDF.head(20)

# Count the number of level 4s
# Zero diagnosis
count0 = 0
count0List = []
count0Levels = []
# One diagnosis
count1 = 0
count1List = []
count1Levels = []
# Two diagnosis
count2 = 0
count2List = []
count2Levels = []
# Three diagnosis
count3 = 0
count3List = []
count3Levels = []
# Four diagnosis
count4 = 0
count4List = []
count4Levels = []

# Set the image cap
capSize = 2

# Perform filtering
for i in range(0,len(currentDF.level)):
	if currentDF.iloc[i,1]==0 and count0 <capSize:
		count0List.append(currentDF.iloc[i,0])
		count0Levels.append(currentDF.iloc[i,1])
		count0 += 1
		# Copy file to Level0 Folder
		copy(correntDR.iloc[i,0],'./Level0/')
	elif currentDF.iloc[i,1]==1 and count1 <capSize:
		count1List.append(currentDF.iloc[i,0])
		count1Levels.append(currentDF.iloc[i,1])
		count1 += 1
		# Copy file to Level0 Folder
		copy(correntDR.iloc[i,0],'./Level1/')
	elif currentDF.iloc[i,1]==2 and count2 <capSize:
		count2List.append(currentDF.iloc[i,0])
		count2Levels.append(currentDF.iloc[i,1])
		count2 += 1
		# Copy file to Level0 Folder
		copy(correntDR.iloc[i,0],'./Level2/')
	elif currentDF.iloc[i,1]==3 and count3 <capSize:
		count3List.append(currentDF.iloc[i,0])
		count3Levels.append(currentDF.iloc[i,1])
		count3 += 1
		# Copy file to Level0 Folder
		copy(correntDR.iloc[i,0],'./Level3/')
	elif currentDF.iloc[i,1]==4 and count4 <capSize:
		count4List.append(currentDF.iloc[i,0])
		count4Levels.append(currentDF.iloc[i,1])
		count4 += 1
		# Copy file to Level0 Folder
		copy(correntDR.iloc[i,0],'./Level4/')

# Create a single data frame
# Lables
labels = count0List + count1List + count2List + count3List + count4List
# Levels
levels = count0Levels + count1Levels + count2Levels + count3Levels + count4Levels


# Create a data frame
selectedImagesDF = pd.DataFrame(np.asarray([labels,levels]).T.tolist(),columns=['image','level'])

# Write it to the csv
selectedImagesDF.to_csv('SelectedImages.csv')

# Plot data
# Plot raw output
# fig = plt.figure(figsize=(20,16))
# ax = fig.add_subplot(2,1,1)
# ax.plot(levels,'o')
# ax.set_title('Selected Image Levels After Filtering', size=24)
# plt.savefig('After_filtering.png')
# plt.show()

