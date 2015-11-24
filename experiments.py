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
# Note: 600 is highest this will go, for 50000 users
def experiment3(delt=0.75, numIter=30, low=100, high=600, step=100, numTrain=30000, numTest=10000):
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
	outdir = "output/"
#	out1 = experiment1()
#	pickle.dump(out1,open("exp1result.pickle","wb"))
#	out2 = experiment2()
#	pickle.dump(out2,open("exp2result.pickle","wb"))
	out3 = experiment3(numIter=3, low=100,high=100)
	pickle.dump(out3,open(outdir+"exp3result.pickle","wb"))

def testmain():
	out1 = experiment1(numIter=1)
	out2 = experiment2(numIter=1, high=5000)
	out3 = experiment3(numIter=1,high=100)

def extractStats(e):
	statsStruct = {}
	for key, val in e.iteritems():
		accuracyArr = np.zeros(len(val))
		errorArr = np.zeros(len(val))
		representative = None
		for i in np.arange(0,len(val)):
			if (i==0):
				representative=(val[i][4],val[i][2],val[i][3])
			accuracyArr[i]=val[i][0]
			errorArr[i]=val[i][1]
		avgAcc = np.average(accuracyArr)
		avgErr = np.average(errorArr)
		stdAcc = np.std(accuracyArr)
		stdErr = np.std(errorArr)
		histAcc = np.histogram(accuracyArr,bins=10)
		histErr = np.histogram(errorArr,bins=10)
		statsStruct[key]=(avgAcc, avgErr, stdAcc, stdErr, histAcc, histErr, representative)
	return statsStruct

def plotRepresentatives(statStruct, label, directory="./images/", extension=".jpg"):
	plt.clf()
	for key, val in statStruct.iteritems():
		accHistLabel = label+'_'+str(key)+"_accuracy"
		errHistLabel = label+'_'+str(key)+"_error"
		repHistLabel = label+'_'+str(key)+"_representative"
		cl.visualize(counts=val[4][0],bins=val[4][1],show=False,title=accHistLabel,xlabel="Accuracy",ylabel="Number of Trials")
		#plt.show()
		plt.savefig(directory+accHistLabel+extension)
		plt.clf()
		cl.visualize(counts=val[5][0],bins=val[5][1],show=False, title=errHistLabel,xlabel="(avg)Error",ylabel="Number of Trials")
		plt.savefig(directory+errHistLabel+extension)
		#plt.show()
		plt.clf()
		cl.visualize(gmm = val[6][0].gmm, counts=val[6][1][0],bins=val[6][1][1],show=False,title=repHistLabel)
		plt.savefig(directory+repHistLabel+extension)
		#plt.show()
		plt.clf()

def printStatistics(statStruct, label, directory="./images/", extension=".csv"):
	filename=directory+label+extension
	fileObj = open(filename,"wb")
	fileObj.write("Parameter, Average Accuracy, Accuracy STD, Average Error, Error STD\n")
	for key, val in statStruct.iteritems():
		fileObj.write(str(key)+','+str(val[0])+','+str(val[2])+','+str(val[1])+','+str(val[3])+'\n')


def loadVisualize():
	exp1 = pickle.load(open("exp1result.pickle"))
	exp2 = pickle.load(open("exp2result.pickle"))
	exp3 = pickle.load(open("exp3result.pickle"))
	exp1struct = extractStats(exp1)
	exp2struct = extractStats(exp2)
	exp3struct = extractStats(exp3)
	plotRepresentatives(exp1struct, "exp1_n=10000")
	plotRepresentatives(exp2struct, "exp2_delt=0.75")
	plotRepresentatives(exp3struct, "exp3_delt=0.75_n=50000")
	printStatistics(exp1struct, "exp1_n=10000")
	printStatistics(exp2struct, "exp2_delt=0.75")
	printStatistics(exp3struct, "exp3_delt=0.75_n=50000")

main()
