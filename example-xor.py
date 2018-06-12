from dee import Dee

# XOR function example

# input data
X = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],	 
]

# output truth
y = [
	0,
	1,
	1,
	0,	 
]

# training the network with two hidden layers of 3 neurons, and 2 output layers
# play with the network and see how the plot changes
network = Dee([3, 3], 2)
network.train(X, y, epochs=2000, learningRate=0.01, batchSize=20)

# predicting the outcomes of a test data: [0, 1]
print(network.predict([
	[0, 1],
])[:, 1])

# and [1, 1]
print(network.predict([
	[1, 1],
])[:, 1])

# network visualizations
network.visualize()
network.plot2D()
network.plotLoss()