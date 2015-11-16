import numpy as np
from sklearn import linear_model
#import matplotlib

def sigmoid(val):
	return 1/(1+np.exp(-val))

#numpy.random.seed(1)
# May want to change range of alpha values slightly
# delta needs to be a column vector
class DataSim:
	def __init__(self,
			 nCovariates = 100, #Number of covarying vars
			 covLower = 0.2, #Lower bound for covariates
			 covUpper = 0.8, #Upper bound for covariates
			 heterLower = 0.4,
			 heterUpper = 0.6,
			 nHeterogenous = 0, #Number of heterogenous effects
			 alphaLower= -0.3, 
			 alphaUpper= 0.3,
			 betaLower= -0.3,
			 betaUpper= 0.3,
			 gammaLower = 0,
			 gammaUpper = 1,
			 delta = np.array([[1.5]]),
			 treatmentEffect = 1,
			 regenerateParams = False,
			 regenerateProbs = False
			 ):

		# Model: treatment probs = sigmoid(X[0:nCovariates]*alpha)
		# OutcomeProbs = sigmoid(X*modelCoeff+treatmentEffect*Z+X[nCovariates:nFeatures]*delta*Z)
		# modelCoeff generated from beta and gamma parameters (for covariates and heter. resp.)
		
		 self.nCovariates = nCovariates
		 self.nHeterogenous = nHeterogenous
		 self.alphaLower = alphaLower
		 self.alphaUpper = alphaUpper
		 self.betaLower = betaLower
		 self.betaUpper = betaUpper
		 self.gammaLower = gammaLower
		 self.gammaUpper = gammaUpper
		 self.covLower = covLower
		 self.covUpper = covUpper
		 self.heterLower = heterLower
		 self.heterUpper = heterUpper
		 self.delta = delta
		 self.treatmentEffect = treatmentEffect
		 self.regeneratePar = regenerateParams
		 self.regenerateProb = regenerateProbs

		 # alphaSetting and modelCoeff are model coefficients
		 self.alphaSetting = np.random.uniform(alphaLower, alphaUpper, nCovariates) #alpha coeff
		 self.featureSize = nCovariates+nHeterogenous
		 self.modelCoeff = np.zeros((self.featureSize,1))
		 self.modelCoeff[0:nCovariates,0]=np.random.uniform(betaLower, betaUpper, nCovariates)
		 self.modelCoeff[nCovariates:self.featureSize,0]=np.random.uniform(gammaLower, gammaUpper, nHeterogenous)

		 #featureProbs consists of all probabilities of observing features in patients
		 self.featureProbs = np.zeros(self.featureSize)
		 self.featureProbs[0:nCovariates] = np.random.uniform(covLower, covUpper, nCovariates)
		 self.featureProbs[nCovariates:self.featureSize] = np.random.uniform(heterLower, heterUpper, nHeterogenous)

	def regenerateParams(self):
		self.alphaSetting = np.random.uniform(self.alphaLower, self.alphaUpper, self.nCovariates) #alpha coeff
		self.modelCoeff[0:self.nCovariates,0]=np.random.uniform(self.betaLower, self.betaUpper, self.nCovariates)
		self.modelCoeff[self.nCovariates:self.featureSize,0]=np.random.uniform(self.gammaLower, self.gammaUpper, self.nHeterogenous)

	def regenerateProbs(self):
		self.featureProbs[0:self.nCovariates] = np.random.uniform(self.covLower, self.covUpper, self.nCovariates)
		self.featureProbs[self.nCovariates:self.featureSize] = np.random.uniform(self.heterLower, self.heterUpper, self.nHeterogenous)

	def getTrueMeans(self):
		# need to generate all binary arrays of length nHeterogenous
		trueMeans = np.zeros(2**self.nHeterogenous)
		if self.nHeterogenous > 0:
			arrangement = np.zeros(self.nHeterogenous,dtype=np.int8)
			for iter in np.arange(0,2**self.nHeterogenous):
				trueMeans[iter]=np.dot(arrangement,self.delta)+self.treatmentEffect
				pos = self.nHeterogenous
				for index in np.arange(0,self.nHeterogenous):
					if (arrangement[index] != 1):
						pos = index+1
						break
				for index in np.arange(0,pos):
					arrangement[index] = 1-arrangement[index]
		else:
			trueMeans[0]=self.treatmentEffect

		trueMeans = np.exp(trueMeans)
		trueMeans.sort()
		return trueMeans


	def generate(self, nPatients):
		if (self.regeneratePar):
			self.regenerateParams()
		if (self.regenerateProb):
			self.regenerateProbs()
		# Generate Patient data
		# there are nPatients patients, each with featureSize features
		# featureSize and featureProbs includes both heterogenous and covarying
		X = np.zeros((nPatients, self.featureSize))
		#for i in np.arange(0,nPatients):
		#	X[i,:] = np.random.binomial(1, self.featureProbs)
		for i in np.arange(0,self.featureSize):
			X[:,i] = np.random.binomial(1,self.featureProbs[i],nPatients)

		# Generate treatment probabilities
		treatmentProbs = sigmoid(np.dot(X[:,0:self.nCovariates],self.alphaSetting))

		# Generate treatment assignments
		Z = np.zeros((nPatients,1))
		Z[:,0]=np.random.binomial(1,treatmentProbs)

		# Use linear model to get outcome probabilities
		linearModel = np.dot(X,self.modelCoeff)+self.treatmentEffect*Z
		if (self.nHeterogenous):
			linearModel += np.dot(X[:,self.nCovariates:self.featureSize],self.delta)*Z
		Yprobs = sigmoid(linearModel)

		# Generate outcomes
		Y = np.random.binomial(1,Yprobs)

		return X, Z, Y

	# Given the means, give class predictions
	#def getTruePredictions(X, means):

	def getTrueClass(self, X):
		means = self.getTrueMeans()
		patientMeans = np.dot(X[:,self.nCovariates:self.featureSize],self.delta)+self.treatmentEffect
		patientMeans = np.exp(patientMeans)
		means = means.reshape((1,means.size))
		tic = np.absolute(patientMeans-means) #hopefully this works
		return np.argmin(tic,axis=1)



# Get the individual odds ratios from the data
def getIndividualOdds(X, Z, Y):
	# Want to train a classifier on Z = 1 and Z = 0
	# Predict Y, get empirical odds ratios
	npatients = X.shape[0]
	indTreated = np.reshape(Z==1,(npatients))
	Xtreated = X[indTreated,:]
	ntreated = Xtreated.shape[0]
	Ytreated = Y[indTreated].reshape((ntreated))

	indUntreated = np.reshape(Z==0,(npatients))
	Xuntreated = X[indUntreated,:]
	nuntreated = Xuntreated.shape[0]
	Yuntreated = Y[indUntreated].reshape((nuntreated))

	treatedClassifier = linear_model.LogisticRegressionCV(cv=5,penalty='l2')
	untreatedClassifier = linear_model.LogisticRegressionCV(cv=5, penalty='l2')
	treatedClassifier.fit(Xtreated, Ytreated)
	untreatedClassifier.fit(Xuntreated, Yuntreated)

	# Later, examine predict_proba in detail
	Z1Predict = treatedClassifier.predict_proba(X)
	#Z1Predict = treatedClassifier.predict(X)
	Z0Predict = untreatedClassifier.predict_proba(X)
	#Z0Predict = untreatedClassifier.predict(X)

	#print Z1Predict
	#print Z0Predict

	individualOdds = (Z1Predict[:,1]/Z1Predict[:,0]) / (Z0Predict[:,1] / Z0Predict[:,0])

	return individualOdds

#thing = DataSim(nPatients = 10000, nCovariates = 100, nHeterogenous=0)
#a,b,c = thing.generate()
#print a.shape
#print a
#print b.shape
#print b
#print c.shape
#print c

# thing = DataSim(nPatients=100, nCovariates = 5, nHeterogenous = 1)
# X, Z, Y = thing.generate()
#print X
#print Z
#print Y
#tac = np.reshape(Z==1, (X.shape[0]))
#print tac
#print X[2:4,:]
#print X[tac,:]

# odds = getIndividualOdds(X,Z,Y)
# print odds.shape
# print odds



#print "Yay! After DataSim definition"        
