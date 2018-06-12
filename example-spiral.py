import numpy as np
import math
from dee import Dee

# a generic function that makes spirals
def spiral(offset=0):
	a = 1
	offsetR = offset /180 * math.pi
	phi = np.arange(0, (3) * np.pi, 0.1)
	x1 = a*phi*np.cos(phi + offsetR)
	x2 = a*phi*np.sin(phi + offsetR)

	dr = (np.diff(x1)**2 + np.diff(x2)**2)**.5 # segment lengths
	r = np.zeros_like(x1)
	r[1:] = np.cumsum(dr) # integrate path
	r_int = np.linspace(0, r.max(), 50) # regular spaced path
	x1 = np.interp(r_int, r, x1) # interpolate
	x2 = np.interp(r_int, r, x2)

	result = np.column_stack((x1,x2))
	result = np.delete(result, 0, axis=0)
	
	return result 

# generating two spirals
spiralA = spiral()
spiralB = spiral(offset=180)

# and concatenating them to form X and y
X = np.concatenate((spiralA, spiralB), axis=0)
y = np.array([0] * len(spiralA) + [1] * len(spiralB))

# training the network
# three hidden layers of 10 neurons and 2 output layers
network = Dee([10, 10, 10], 2)
network.train(X, y, epochs=15000, learningRate=0.001, batchSize=20)

# visualization
network.visualize()
network.plot2D()
network.plotLoss()

print("Play with the network parameters to get a better result")