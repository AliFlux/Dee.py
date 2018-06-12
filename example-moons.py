import sklearn
import sklearn.datasets
from dee import Dee

# importing moon dataset with 25% noise
# NOTE: the result will always be different due to random noise
X, y = sklearn.datasets.make_moons(200, noise=0.25)

# training on three hidden layers with 5,2,5 neurons
network = Dee([5, 2, 5], 2)
network.train(X, y, epochs=5000, learningRate=0.01, batchSize=20)

# visualization
network.visualize()
network.plot2D()
network.plotLoss()
