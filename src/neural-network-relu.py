import neuralnetworksA2 as nn
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt

class NeuralNetworkReLU(nn.NeuralNetwork):
	def __init__(self, ni, nh, no):
		super(NeuralNetworkReLU, self).__init__(ni, nh, no)

	def activation(self, weighted_sum):
		return np.maximum(0, weighted_sum)

	def activationDerivative(self, activation_value):
		actDer = np.copy(activation_value)
		actDer[actDer <= 0] = 0
		actDer[actDer > 0] = 1
		return actDer

def partition(X, T, fraction, shuffle):
    nRows = X.shape[0]
    nTrain = int(round(fraction*nRows)) 
    nTest = nRows - nTrain

    rows = np.arange(nRows)

    if(shuffle == True):
        np.random.shuffle(rows)

    trainIndices = rows[:nTrain]
    testIndices = rows[nTrain:]

    Xtrain = X[trainIndices, :]
    Ttrain = T[trainIndices, :]
    Xtest = X[testIndices, :]
    Ttest = T[testIndices, :]
    
    return Xtrain, Ttrain, Xtest, Ttest

def rmse(A, B):
    return np.sqrt(np.mean((A - B)**2))

if __name__ == '__main__':
	
	# Load the csv data.
	dframe = pd.read_csv('energydata_complete.csv', sep=',',header=None)
	# Filter out required columns.
	#dframe = dframe.drop(dframe.columns[[0, -2, -1]], axis=1)

	# Get target.
	Td = dframe.iloc[1:, [1]]
	Td = Td.as_matrix()
	T = Td.astype(float)

	# Get input.
	Xd = dframe.iloc[1:, 2:-2]
	Xd = Xd.as_matrix()
	X = Xd.astype(float)
	
	# Smaller dataset
	X = np.arange(5).reshape((-1,1))
	T = np.sin(X)

	# Comparision
	#hiddenLayers = [[u]*nl for u in [1, 2, 5, 10, 50] for nl in [1, 2, 3, 4, 5, 10]]
	hiddenLayers = [[1], [1,1], [5], [5,5], [10]]
	tanHlist = []
	ReLUlist = []
	for actFun in [nn.NeuralNetwork, NeuralNetworkReLU]:
		for hidden in hiddenLayers:
			# Create list for storing RMSE.
			rmseTrainList = []
			rmseTestList = [] 
			for i in range(10):
				Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8, shuffle = False)
				nnet = actFun(Xtrain.shape[1], hidden, Ttrain.shape[1])
				nnet.train(Xtrain, Ttrain, 100)
				rmseTrain = rmse(Ttrain, nnet.use(Xtrain))
				rmseTest = rmse(Ttest, nnet.use(Xtest))
				rmseTrainList.append(rmseTrain)
				rmseTestList.append(rmseTest)
			rmseTrainMean = sum(rmseTrainList)/len(rmseTrainList)
			rmseTestMean = sum(rmseTestList)/len(rmseTestList)
			if(actFun == nn.NeuralNetwork):
				tanHlist.append([hidden, rmseTrainMean, rmseTestMean])
			else:
				ReLUlist.append([hidden, rmseTrainMean, rmseTestMean])
	
	print("\n\n 1. tanH:")
	tanHlist = pd.DataFrame(tanHlist)
	pp.pprint(tanHlist)
	
	print("\n\n 2. ReLUlist:")
	ReLUlist = pd.DataFrame(ReLUlist)
	pp.pprint(ReLUlist)

	plt.figure(figsize = (10, 10))
	plt.plot(tanHlist.values[:, 1], 'b', label = 'tanH Train RMSE')
	plt.plot(tanHlist.values[:, 2], 'g', label = 'tanH Test RMSE')
	plt.plot(ReLUlist.values[:, 1], 'm', label = 'ReLU Train RMSE')
	plt.plot(ReLUlist.values[:, 2], 'k', label = 'ReLU Test RMSE')
	#plt.plot(tanHlist.values[:, 1:], 'o-')
	#plt.plot(ReLUlist.values[:, 1:], 'o-')
	plt.legend(('tanh Train RMSE', 'tanh Test RMSE', 'ReLU Train RMSE', 'ReLU Test RMSE'))	
	plt.xticks(range(tanHlist.shape[0]), hiddenLayers, rotation=30, horizontalalignment='right')
	plt.grid(True)
	plt.show()


