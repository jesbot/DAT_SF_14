# evaluatePerformance.py
#
# Purpose: Goes throught the output of a deep learning run and 
# categorizes the output. 
#  
# Author: Jose Solomon

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# To find the correct label for the image
import re

# Get an encoder for performance metrics
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# To determine accuracy and confusion matrix
from sklearn.metrics import (confusion_matrix, accuracy_score,classification_report, roc_curve)

# Save to figure?
SAVEFIGURE = True

# Let's begin by importing the data

# True/False data set: individual has DR, individual does not have DR
# dnnTrain = pd.read_csv('GoogLeNet_Train_Results_TF_80_VS_10_TS_10.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen','Cert_2']))
# dnnTrain = pd.read_csv('AlexNet_Train_Results_1000_TS_90_VS_5_TS_5.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen_1','Cert_2',\
# 		'Not_Chosen_2','Cert_3','Not_Chosen_3','Cert_4','Not_Chosen_4','Cert_5']))
# dnnTrain = pd.read_csv('GoogLeNet_200_Prediction_Level0_Level4.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen','Cert_2']))
# dnnTrain = pd.read_csv('GoogLeNet_200_Results_Level0_Level3_4.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen','Cert_2']))
# dnnTrain = pd.read_csv('GoogLeNet_Train_Results_1000_TS_90_VS_5_TS_5.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen_1','Cert_2',\
# 		'Not_Chosen_2','Cert_3','Not_Chosen_3','Cert_4','Not_Chosen_4','Cert_5']))
# dnnTrain = pd.read_csv('AlexNet_Val_Results_Bias.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen_1','Cert_2',\
# 		'Not_Chosen_2','Cert_3','Not_Chosen_3','Cert_4','Not_Chosen_4','Cert_5']))
# dnnTrain = pd.read_csv('AlexNet_Mod_w_Bias.csv',header=False,\
# 	names=(['Image_Class','Classified','Cert_1','Not_Chosen_1','Cert_2',\
# 		'Not_Chosen_2','Cert_3','Not_Chosen_3','Cert_4','Not_Chosen_4','Cert_5']))
dnnTrain = pd.read_csv('AlexNet_Custom2.csv',header=False,\
	names=(['Image_Class','Classified','Cert_1','Not_Chosen_1','Cert_2',\
		'Not_Chosen_2','Cert_3','Not_Chosen_3','Cert_4','Not_Chosen_4','Cert_5']))
# print dnnTrain.head()


# Number of entries
nEntries = dnnTrain.shape[0]
# Store the correct label
correctL = []
# Parse the image name entry to determine the correct label
for i in range(0,nEntries):
	imageName = dnnTrain.Image_Class[i]
	# correct = re.search('Image_True_False/(.+?)/',imageName).group(1)
	# correct = re.search('ImageDataSet_1000_PNG/(.+?)_PNG/',imageName).group(1)
	# correct = re.search('TF_Level0_Level4/(.+?)_/',imageName).group(1)
	# correct = re.search('TF_Level0_Level3_4/(.+?)_/',imageName).group(1)
	correct = re.search('ImageSet/(.+?)_PNG/',imageName).group(1)
	correctL.append(correct)

# SAve the result as an array	
correctLabels = np.array(correctL)

# Now add the correct label to the data frame
dnnTrain['CorrectLabel'] = correctLabels

# # Just checking something
print dnnTrain.head(10)

# Create a dataframe of tokenized elements
labelDF = pd.concat(([dnnTrain['CorrectLabel'], dnnTrain['Classified'], dnnTrain['Cert_1']]),\
	axis=1,keys=('CorrectLabel','NNLabel','Prob'))

# Take the percentages in 'Cert1' and convert them to floats
labelDF['Prob'] = labelDF['Prob'].str.rstrip('%').astype('float64')/100
print labelDF.head(10)

# Tokenize it
le.fit(labelDF['CorrectLabel'])	
labelDF['CorrectLabel'] = le.transform(labelDF['CorrectLabel'])
le.fit(labelDF['NNLabel'])	
labelDF['NNLabel'] = le.transform(labelDF['NNLabel'])

## Performance metrics
# Accuracy:
m = accuracy_score(labelDF['CorrectLabel'],labelDF['NNLabel'])
print m

# Create confusion matrix
cm = confusion_matrix(labelDF['CorrectLabel'],labelDF['NNLabel'])
print cm

# Plot confusion matrix
# target_names=['DR -', 'DR +']   
# target_names=['DR 0', 'DR 4'] 
# target_names=['DR 0', 'DR 3 & 4'] 
target_names=['DR 0', 'DR 4','DR 2','DR 3', 'DR 1']
def plot_confusion_matrix(cm, title='Confusion Matrix Full Classifier', cmap=plt.cm.Reds):
   	plt.imshow(cm, interpolation='nearest', cmap=cmap)
   	plt.title(title)
   	plt.colorbar()
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names)
	plt.yticks(tick_marks, target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.gcf().tight_layout()
	# Save the figure
	if SAVEFIGURE:
		plt.savefig(title)
	plt.show()
plot_confusion_matrix(cm)

# Print a full classification report with labels
print classification_report(labelDF['CorrectLabel'],labelDF['NNLabel'],\
	target_names=target_names)

# Create a normalized confusion matrix
cmNormalized = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
print cmNormalized
plot_confusion_matrix(cmNormalized,'Normalized Confusion Matrix Full Classifier')

fig = plt.figure()
plt.scatter(labelDF['CorrectLabel'],labelDF['Prob'],color='blue')
title = 'Certainty of Label Assignment - Full Classification'
fig.suptitle(title)
plt.xlabel('Classification Label')
plt.ylabel('Confidence')
plt.xticks([0, 1, 2, 3, 4])
plt.ylim([0,.6])

plt.show()
if SAVEFIGURE:
	fig.savefig(title)
