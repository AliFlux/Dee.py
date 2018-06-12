import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# Helper class that contians functions to visualize the dee.py network
class VisualizeDee:
	
	# helps in printing progress of the training
	@staticmethod
	def printProgress(value, total, length = 20, threshold = 0.1, extras = ""):
		
		if value % (total * threshold) != 0:
			return
		
		sys.stdout.write('\r')
		
		barLength = math.ceil(value/total * length)
		progressString = "-" * barLength + ">" + " " * (length - barLength)
		sys.stdout.write("[" + progressString + "] " + str(round(value/total * 100)) + "% " + extras)
		
		sys.stdout.flush()
	
	# visualization of the neural network in form of a graph
	# based on https://gist.github.com/craffel/2d727968c3aaebd10359
	def visualize(dee):
		
		# print this to see the nodes in each layer
		layer_sizes = np.concatenate(([np.shape(dee.X)[1]], dee.hiddenLayers, [dee.outputNodes]))
		
		# making the plot
		fig = plt.figure(figsize=(len(layer_sizes) * 2.4, np.max(layer_sizes) * 1.2))
		fig.clear()
		fig.set_tight_layout(False)
		ax = fig.gca()
		ax.margins(0.1)
		ax.axis('off')
		left, right, bottom, top = 0.1, 0.9, 0.3, 0.7
		
		v_spacing = (top - bottom)/float(max(layer_sizes))
		h_spacing = (right - left)/float(len(layer_sizes) - 1)
		
		# print this to see colors of each layer
		colors = np.concatenate((["#2ecc71"], ["#3498db"] * int(len(layer_sizes)-2), ["#e74c3c"]))
		
		# drawing nodes/neurons/circles of each layer
		for n, layer_size in enumerate(layer_sizes):
			layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
			for m in range(layer_size):
				ax.plot(n*h_spacing + left, layer_top - m*v_spacing,'o',color=colors[n],fillstyle='full',markersize=40, zorder=4)
				ax.plot(n*h_spacing + left, layer_top - m*v_spacing,'o',color='white',fillstyle='full',markersize=45, zorder=3)
		
		# drawing lines/synapses between nodes
		for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
			layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
			
			weights = dee.W[n]
			minWeight, maxWeight = np.min(weights), np.max(weights)
			
			for m in range(layer_size_a):
				for o in range(layer_size_b):
					lineWidth = (weights[m, o]-minWeight)/(maxWeight-minWeight) * 5
					line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
									  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], color='#778ca3', linewidth=lineWidth)
					ax.add_artist(line)
	
	# plots the result of the network in form of a 2D chart
	# based on http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
	def plot2D(dee, colorMap = plt.cm.rainbow, resolution = 0.01, discrete = False, yColumn = 1):
		
		# making the plot
		fig = plt.figure(figsize=(10, 8))
		fig.clear()
		fig.set_tight_layout(False)
		ax = fig.gca()
		
		# finding min and maxes of both axes
		x_min, x_max = dee.X[:, 0].min() - .5, dee.X[:, 0].max() + .5
		y_min, y_max = dee.X[:, 1].min() - .5, dee.X[:, 1].max() + .5
		
		# making a mesh grid of each pixel/point
		xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
	
		# running prediction on each point
		Z = dee.predict(np.c_[xx.ravel(), yy.ravel()])[:,yColumn]
		
		# if discrete, then round the results to 0 or 1
		if discrete:
			Z = np.round(Z)
		
		# reshape it to a matrix
		Z = Z.reshape(xx.shape)
		
		# plot it!
		ax.contourf(xx, yy, Z, cmap=colorMap)
		ax.scatter(dee.X[:, 0], dee.X[:, 1], c=dee.y, cmap=colorMap)
		
		
	def plotLoss(dee, color = 'r'):
		
		# plots the loss of the training
		fig = plt.figure(figsize=(10, 8))
		fig.clear()
		fig.set_tight_layout(False)
		ax = fig.gca()
		ax.plot(dee.loss, color = color)