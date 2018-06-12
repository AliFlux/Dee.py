import numpy as np
import matplotlib.pyplot as plt
from visualize_dee import VisualizeDee

# Dee.py core neural network class
class Dee:
	
	# initialization constructor
	def __init__(self, hiddenLayers, outputNodes=1):
		self.hiddenLayers = hiddenLayers
		self.outputNodes = outputNodes
		self.datasetLength = 0
		self.W = []
		self.b = []
	
	# standalone prediction function that can give output of any test input
	def predict(self, x):
		
		# basically forward propogation happening here
		zAll = []
		aAll = []
	
		zAll.append([])
		aAll.append(np.array(x))
	
		WAll = self.W
		bAll = self.b
	
		for j in range(0, len(self.W)):
			W = WAll[j]
			b = bAll[j]
	
			inputData = aAll[j]
			
			z = inputData.dot(W) + b
			
			# modify to change activation function
			a = np.tanh(z)
	
			zAll.append(z)
			aAll.append(a)
		
		# probabilities of scores
		exp_scores = np.exp(zAll[len(zAll) - 1])
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		
		return probs
	
	# a helper function that divides dataset into batches
	@staticmethod
	def batch(iterable, n=1):
		l = len(iterable)
		
		result = []
		for ndx in range(0, l, n):
			result.append(iterable[ndx:min(ndx + n, l)])
			
		return result
	
	# the core training function
	def train(self, X, y, epochs = 10000, learningRate=0.1, batchSize = 5):
		np.random.seed(0)
		
		X = np.array(X)
		y = np.array(y)
		
		self.X = X
		self.y = y
		self.datasetLength = len(X)
		numHiddenLayers = len(self.hiddenLayers)
		numProcessingLayers = numHiddenLayers + 1
	
		WAll = []
		bAll = []
		
		# generate dimensions for each layer
		dimensions = [np.shape(X)[1]]
		
		for numNodes in self.hiddenLayers:
			dimensions.append(numNodes)
			
		dimensions.append(self.outputNodes)
		
		# initialize to random weights and biases
		for i in range(0, numProcessingLayers):
			WAll.append(np.random.randn(dimensions[i], dimensions[i+1]) / np.sqrt(dimensions[i]))
			bAll.append(np.zeros((1, dimensions[i+1])))
		
		self.W = []
		self.b = []
		self.loss = []
		
		# generate batches of the data
		XBatches = Dee.batch(X, batchSize)
		yBatches = Dee.batch(y, batchSize)
		
		for i in range(0, epochs):
			
			epochActivations = []
			
			for x in range(0, len(XBatches)):
				
				Xbatch = np.array(XBatches[x])
				yBatch = np.array(yBatches[x])
				thisBatchSize = len(Xbatch)
				
				zAll = []
				aAll = []
		
				zAll.append([])
				aAll.append(Xbatch)
		
				# forward propogate the network
				for j in range(0, numProcessingLayers):
					W = WAll[j]
					b = bAll[j]
		
					inputData = aAll[j]
					
					# crunch the numbers and activate the function
					z = inputData.dot(W) + b
					a = np.tanh(z)
		
					zAll.append(z)
					aAll.append(a)
				
				# calculate errors and probabilities of last layer
				lastError = zAll[len(zAll) - 1]
				lastActivation = aAll[len(aAll) - 1]
				
				for item in lastActivation:
					epochActivations.append(item)
				
				exp_scores = np.exp(lastError)
				probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
				
				# Backpropagate through the network
				deltaL = probs
				deltaL[range(thisBatchSize), yBatch] -= 1
				
				dWAll = []
				dbAll = []
		
				lastDelta = deltaL
				
				# reverse loop through the layers
				for j in reversed(range(0, numProcessingLayers)):
					
					# calculate & add delta weight and biases
					dWAll.insert(0, (aAll[j].T).dot(lastDelta))
					dbAll.insert(0, np.sum(lastDelta, axis=0, keepdims=(j == 0)))
		
					lastDelta = lastDelta.dot(WAll[j].T) * (1 - np.power(aAll[j], 2))
		
				# apply the weights to the model
				# the delta is multiplied to the learning rate and adjusted to W/b
				for j in range(0, numProcessingLayers):
					WAll[j] += -learningRate * dWAll[j]
					bAll[j] += -learningRate * dbAll[j]
			
			# save the weights and biases
			self.W = WAll;
			self.b = bAll;
			
			# calculate the errors of this epoch
			epochActivations = np.array(epochActivations)
			
			validActivations = epochActivations[range(self.datasetLength), self.y]
			
			# via squared error formula
			lossNum = 1 - np.mean(np.square(validActivations))
			self.loss.append(lossNum)
			
			# print the progress bar
			VisualizeDee.printProgress(i, epochs, 30, 0.01, "Epoch: " + str(i) + " Loss: " + str(lossNum))
		print("\n")
	
	# helper functions that visualize data
	# implemented in dee.py
	def visualize(self):
		VisualizeDee.visualize(self)
			
	def plot2D(self, colorMap = plt.cm.rainbow, resolution = 0.01, discrete = False, yColumn = 1):
		VisualizeDee.plot2D(self, colorMap, resolution, discrete, yColumn)
			
	def plotLoss(self, color = 'r'):
		VisualizeDee.plotLoss(self, color)
	

		
		

		
		