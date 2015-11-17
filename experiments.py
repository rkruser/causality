import model as mdl
import dataClass as dc
import clusterVisualize as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import mixture
import pickle

# One heterogenous variable
# Iterate over deltas
# Regenerate model parameters
def experiment1(numIter=30, numTrain=10000, numTest=10000):
	deltaDict = {}
	for delt in np.arange(-2.0,2.5,0.5):
		datgen = dc.DataSim(nCovariates=100,nHeterogenous=1,delta=np.array([[delt]]))
		datmodel = mdl.simModel(generator = datgen)
		data = []
		for i in np.arange(0,numIter):
			Xtrain, Ztrain, Ytrain = datmodel.generate(numTrain)
			datmodel.train(Xtrain, Ztrain, Ytrain)
			Xtest, Ztest, Ytest = datmodel.generate(numTest)
			accuracy, h1, h2 = datmodel.test(Xtest,Ztest,Ytest)
			data.append((accuracy,h1,h2))
			datmodel.regenerate() #re-generate the coefficients
		deltaDict[delt]=data

	return deltaDict


# 
#def experiment2():

out = experiment1(2,10000,5000)
pickle.dump(out,open("testout.pickle","wb"))
print out
