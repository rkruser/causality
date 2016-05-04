#cluster and visualize
import dataClass as dat
import numpy as np
from sklearn import mixture
import scipy.stats as stats
import matplotlib.pyplot as plt

# Since the python gmm does not sort its means, we need to do that ourselves
# otherwise, the given predictions will not match the true classes
def getSortedPredictions(predictions, means):
	#put means next to indices, sort, replace predictions
	means = means.reshape(means.size)
	indices = np.argsort(means)
	newPredictions = np.zeros(predictions.size)
	for ind in np.arange(0,predictions.size):
		newPredictions[ind] = indices[predictions[ind]]
	return newPredictions.astype(np.int8)

# logOdds needs to already be in correct shape (length, 1)
def getGmm(logOdds, numComponents=2, BIC=False, BIC_Complexity_Max=8):
	logOdds = logOdds.reshape((logOdds.size,1))
	if BIC:
		smallest_BIC = 1000000000
		bestGm = mixture.GMM(n_components=numComponents) #placeholder default
		for nComponents in np.arange(1,BIC_Complexity_Max+1):
			gm = mixture.GMM(n_components=nComponents)
			gm.fit(logOdds)
			thisBic = gm.bic(logOdds)
			#print "BIC for iter ", nComponents, " = ", thisBic
			if (thisBic < smallest_BIC):
				bestGm = gm
				smallest_BIC = thisBic
		#predictions = bestGm.predict(logOdds)
		return bestGm
		#return bestGm, getSortedPredictions(predictions, gm.means_)

	else:
		gm = mixture.GMM(n_components=numComponents)
		gm.fit(logOdds)
		#predictions = gm.predict(logOdds)
		return gm
		#return gm, getSortedPredictions(predictions, gm.means_)

def visualizeGmm(gmm, logOdds):
	plt.hist(logOdds,bins=50, normed=True)
	plt.title("Log Odds Ratio Histogram")
	plt.xlabel("Log Odds Ratio")
	plt.ylabel("Frequency")
	means = gmm.means_
	variances = gmm.covars_
	x_plot = np.linspace(min(logOdds),max(logOdds),200)
	for m in np.arange(0,means.size):
		mu = means[m,0]
		sigma = variances[m,0]**0.5
		rv = stats.norm(mu,sigma)
		plt.plot(x_plot, rv.pdf(x_plot))
	plt.show()

def visualize(gmm=None, counts=None, bins=None, show=True,
 title="Log Odds Ratio Histogram",
 xlabel="Log Odds Ratio",
 ylabel="Frequency"):
	#plt.hist(counts,bins=bins, normed=True)
	width = bins[1]-bins[0]
	plt.bar(bins[:bins.size-1], counts, width=width) #How do I plot a pre-made histogram?
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	if (gmm is not None):
		means = gmm.means_
		variances = gmm.covars_
		x_plot = np.linspace(min(bins),max(bins),200)
		for m in np.arange(0,means.size):
			mu = means[m,0]
			sigma = variances[m,0]**0.5
			rv = stats.norm(mu,sigma)
			plt.plot(x_plot, rv.pdf(x_plot))
	if (show):
		plt.show()

def getError(unsorted_Means, sorted_True_Means):
	means = unsorted_Means.reshape(unsorted_Means.size)
	means.sort()
	#print "Learned means", means
	#print "True means", sorted_True_Means
	errs = np.absolute(means-sorted_True_Means)/sorted_True_Means
	return errs, np.average(errs), np.std(errs)

def getAccuracy(trueClass, predictedClass):
	correct = np.sum(trueClass==predictedClass)
	return np.float64(correct) / trueClass.size

# np.random.seed(4)

# # delta needs to be a column vector
# n=3
# nPatients = 100000
# ourdelta = np.array([[1.5],[3],[4.5]])#,[0.5],[-0.25],[-2],[3],[5]])
# datagen = dat.DataSim(nCovariates = 100, nHeterogenous = n,
# 						treatmentEffect=1, delta=ourdelta)
# means = datagen.getTrueMeans()
# #print "True means:", means
# X, Z, Y = datagen.generate(nPatients)
# oddsRatios = dat.getIndividualOdds(X,Z,Y)
# logOdds = np.log(oddsRatios)
# logOdds = logOdds.reshape((nPatients,1))

# #gm, predictions = getGmm(logOdds, numComponents = 2**n)
# gm = getGmm(logOdds, BIC = True)
# predictions = gm.predict(logOdds)
# sortedPredictions = getSortedPredictions(predictions, gm.means_)
# trueClasses = datagen.getTrueClass(X)

# print "Number of components", gm.means_.size

# if gm.means_.size == means.size:
# 	# Don't forget to exp the log means!
# 	errs, avgErr, stdErr = getError(np.exp(gm.means_), means)
# 	print "Errors:", errs
# 	print "average Error:", avgErr
# 	print "std err", stdErr
# 	print "predictions", predictions[0:10]
# 	print "true classes", trueClasses[0:10]
# 	print "accuracy on predictions", getAccuracy(trueClasses, predictions)
# 	print "accuracy on sorted predictions", getAccuracy(trueClasses, sortedPredictions)

# print "true means", means
# print "gm means", np.exp(gm.means_)
# visualizeGmm(gm, logOdds)
#print logOdds[0:30]


#visualizeGmm(gm, logOdds)

# Using log odd ratios is a smashing success!
# Next step: Get true means
# Figure out clustering
# Then do experiments! Some with bayesian information criterion, some not!

# Functions to write:
# function that returns the GMM model and fit
# Function that plots the gmm model
# function that saves the gmm model and other stuff to files
# Function that loads stuff from files and plots it
# Function that predicts class, and gets true class, of patients

# Need to join true means to learned means, sort by learned means
