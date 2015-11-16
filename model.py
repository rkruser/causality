#Simulations
import dataClass as dc
import clusterVisualize as cl
import numpy as np
from sklearn import linear_model

class simModel:
	def __init__(self, generator = dc.DataSim(nCovariates = 100,nHeterogenous=1),
		BIC=False):
		self.dataGen = generator
		self.treatedClassif = linear_model.LogisticRegressionCV(cv=5,penalty='l2')
		self.untreatedClassif = linear_model.LogisticRegressionCV(cv=5,penalty='l2')
		self.gmm = None
		self.bic = BIC
	
	def generate(self,nPatients):
		return self.dataGen.generate(nPatients)

	def train(self, X, Z, Y):
		print "Train"
		npatients = X.shape[0]
		indTreated = np.reshape(Z==1,(npatients))
		Xtreated = X[indTreated,:]
		ntreated = Xtreated.shape[0]
		Ytreated = Y[indTreated].reshape((ntreated))

		indUntreated = np.reshape(Z==0,(npatients))
		Xuntreated = X[indUntreated,:]
		nuntreated = Xuntreated.shape[0]
		Yuntreated = Y[indUntreated].reshape((nuntreated))

		self.treatedClassif.fit(Xtreated, Ytreated)
		self.untreatedClassif.fit(Xuntreated, Yuntreated)

		# Later, examine predict_proba in detail
		Z1Predict = self.treatedClassif.predict_proba(X)

		Z0Predict = self.untreatedClassif.predict_proba(X)
		#Z0Predict = untreatedClassifier.predict(X)

		#print Z1Predict
		#print Z0Predict
		print "Gmm train"
		individualOdds = (Z1Predict[:,1]/Z1Predict[:,0]) / (Z0Predict[:,1] / Z0Predict[:,0])
		logOdds = np.log(individualOdds)
		self.gmm = cl.getGmm(logOdds, numComponents=2**self.dataGen.nHeterogenous,BIC=self.bic)

	def getMeansError(self):
		trueMeans = self.dataGen.getTrueMeans()
		if self.gmm.means_.size == trueMeans.size:
			return cl.getError(np.exp(self.gmm.means_), trueMeans)
		else:
			return -1, -1, -1

	def test(self, X, Z, Y):
		print "Test"
		Z1Predict = self.treatedClassif.predict_proba(X)
		Z0Predict = self.untreatedClassif.predict_proba(X)
		individualOdds = (Z1Predict[:,1]/Z1Predict[:,0]) / (Z0Predict[:,1] / Z0Predict[:,0])
		individualOdds = individualOdds.reshape((individualOdds.size,1))
		logOdds = np.log(individualOdds)
		predictions = self.gmm.predict(logOdds)
		sortedPredictions = cl.getSortedPredictions(predictions, self.gmm.means_)
		trueClasses = self.dataGen.getTrueClass(X) #Assumes X was created from same model
		accuracy = cl.getAccuracy(trueClasses, sortedPredictions)

		return accuracy, np.histogram(logOdds,bins=50,density=True), np.histogram(individualOdds,bins=50,density=True)

np.random.seed(19)

genrate = dc.DataSim(nCovariates = 100, nHeterogenous=2, delta=np.array([[2],[4]]))
thing = simModel(generator=genrate)
Xtrain,Ztrain,Ytrain = thing.generate(30000)
thing.train(Xtrain,Ztrain,Ytrain)
Xtest, Ztest, Ytest = thing.generate(5000)
accuracy, h1, h2 = thing.test(Xtest,Ztest,Ytest)
print "accuracy: ", accuracy
cl.visualize(thing.gmm, h1[0], h1[1])
#cl.visualize(thing.gmm, h2[0], h2[1])