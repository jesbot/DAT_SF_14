'''
reLu.py

Purpose:
Plot the ReLu activation function.

jose.e.solomon@gmail.com
'''

import numpy as np
import matplotlib.pyplot as plt

# Define the reLu function
xRange = range(-4,5,1)
print xRange
output = np.zeros(len(xRange))
c = 0
for x in xRange:
	output[c] = max(0,x)
	c +=1

fig, ax = plt.subplots(nrows=1, ncols= 1)
ax.plot(xRange,output,color='red')
ax.set_title('ReLu Activatrereon -- max(0,x)')

plt.show()

