import model as mdl
import dataClass as dc
import clusterVisualize as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import mixture
import pickle

np.random.seed(1)

# One heterogenous variable
# Iterate over deltas
# Regenerate model parameters
def experiment1(numIter=30, numTrain=10000, numTest=10000):
	print "experiment 1"
	deltaDict = {}
	for delt in np.arange(-3.0,3.5,0.5):
		print "  delta=",delt
		datgen = dc.DataSim(nCovariates=100,nHeterogenous=1,delta=np.array([[delt]]))
		datmodel = mdl.simModel(generator = datgen)
		data = []
		for i in np.arange(0,numIter):
			print "    iter=",i
			Xtrain, Ztrain, Ytrain = datmodel.generate(numTrain)
			datmodel.train(Xtrain, Ztrain, Ytrain)
			Xtest, Ztest, Ytest = datmodel.generate(numTest)
			accuracy, h1, h2 = datmodel.test(Xtest,Ztest,Ytest)
			errs,avgerrs,stderrs = datmodel.getMeansError()
			data.append((accuracy,avgerrs,h1,h2,datmodel))
			datmodel.regenerate() #re-generate the coefficients
		deltaDict[delt]=data

	return deltaDict

# Sample size
def experiment2(delt=0.75, numIter=30, low=5000, high = 100000, step=5000, numTest=10000):
	print "experiment 2"
	datgen = dc.DataSim(nCovariates=100,nHeterogenous=1,delta=np.array([[delt]]))
	datmodel=mdl.simModel(generator=datgen)
	trainDict = {}
	for numTrain in np.arange(low,high+step,step):
		print "  numTrain=",numTrain	
		data=[]
		for i in np.arange(0,numIter):
			print "    iter=",i
			Xtrain, Ztrain, Ytrain = datmodel.generate(numTrain)
			datmodel.train(Xtrain, Ztrain, Ytrain)
			Xtest, Ztest, Ytest = datmodel.generate(numTest)
			accuracy, h1, h2 = datmodel.test(Xtest,Ztest,Ytest)
			errs,avgerrs,stderrs = datmodel.getMeansError()
			data.append((accuracy,avgerrs,h1,h2,datmodel))
			datmodel.regenerate() #re-generate the coefficients
		trainDict[numTrain]=data
	return trainDict

# number of covariates
def experiment3(delt=0.75, numIter=30, low=100, high=2000, step=100, numTrain=50000, numTest=10000):
	print "experiment 3"
	dimDict = {}
	for numCovs in np.arange(low,high+step,step):
		print "  numCovs=", numCovs
		datgen = dc.DataSim(nCovariates=numCovs,nHeterogenous=1,delta=np.array([[delt]]))
		datmodel=mdl.simModel(generator=datgen)
		data=[]
		for i in np.arange(0,numIter):
			print "    iter=",i
			Xtrain, Ztrain, Ytrain = datmodel.generate(numTrain)
			datmodel.train(Xtrain, Ztrain, Ytrain)
			Xtest, Ztest, Ytest = datmodel.generate(numTest)
			accuracy, h1, h2 = datmodel.test(Xtest,Ztest,Ytest)
			errs,avgerrs,stderrs = datmodel.getMeansError()
			data.append((accuracy,avgerrs,h1,h2,datmodel))
			datmodel.regenerate() #re-generate the coefficients
		dimDict[numCovs]=data
	return dimDict

def main():
#	out1 = experiment1()
#	pickle.dump(out1,open("exp1result.pickle","wb"))
#	out2 = experiment2()
#	pickle.dump(out2,open("exp2result.pickle","wb"))
	out3 = experiment3(low=100,high=500)
	pickle.dump(out3,open("exp3result.pickle","wb"))

def testmain():
	out1 = experiment1(numIter=1)
	out2 = experiment2(numIter=1, high=5000)
	out3 = experiment3(numIter=1,high=100)

main()
