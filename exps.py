import model as mdl
import dataClass as dc
import clusterVisualize as cl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import mixture
import pickle

EXP_1 = 1 # Number of patients vs dimension
EXP_1_MAX_PATIENTS = 50000
EXP_1_PATIENT_STEP = 5000
EXP_1_MAX_DIMENSION = 2000
EXP_1_DIM_STEP = 100
EXP_1_NSPARSE = 0
EXP_1_NHETER = 1
EXP_1_HETERMEANS = np.array([[1.5]]) #umm, maybe?
EXP_1_NITERS = 30
EXP_1_NTEST = 2000 #?
EXP_1_BIC = False


EXP_2 = 2 # Number of patients vs Sparsity
EXP_2_MAX_PATIENTS = 50000
EXP_2_PATIENT_STEP = 5000
EXP_2_MAX_SPARSITY = 90
EXP_2_SPARSE_STEP = 10
EXP_2_NDIMS = 500
EXP_2_NHETER = 1
EXP_2_HETERMEANS = np.array([[1.5]]) #umm, maybe?
EXP_2_NITERS = 30
EXP_2_NTEST = 2000 #?
EXP_2_BIC = False

EXP_3 = 3 # Sparsity vs dimension
EXP_3_MAX_DIMENSION = 2000
EXP_3_DIM_STEP = 200
EXP_3_MAX_SPARSITY = 90
EXP_3_SPARSE_STEP = 10
EXP_3_NPATIENTS = 30000
EXP_3_NHETER = 1
EXP_3_HETERMEANS = np.array([[1.5]]) #umm, maybe?
EXP_3_NITERS = 30
EXP_3_NTEST = 2000 #?
EXP_3_BIC = False

EXP_4 = 4 # Number of subgroups vs dimension
EXP_4_MAX_DIMENSION = 2000
EXP_4_DIM_STEP = 200
EXP_4_MAX_HETER = 4
EXP_4_HETER_STEP = 1
EXP_4_NPATIENTS = 30000
EXP_4_NSPARSE = 0
EXP_4_HETERMEANS = np.array([[1.5],[2],[3],[4]]) #should be all means, here
EXP_4_NITERS = 30
EXP_4_NTEST = 2000 #?
EXP_4_BIC = True #?

EXP_5 = 5 # Number of means vs separation of means
EXP_5_MAX_MEAN = 3
EXP_5_MEAN_STEP = 1 #?
EXP_5_MAX_HETER = 4
EXP_5_HETER_STEP = 1
EXP_5_NSPARSE = 0
EXP_5_NDIMS = 500
EXP_5_NPATIENTS = 30000
EXP_5_NHETER = 1
EXP_5_HETERMEANS = np.array([[1.5]]) #umm, maybe?
EXP_5_NITERS = 30
EXP_5_NTEST = 2000 #?
EXP_5_BIC = True #?



def generateExperiments():
    # Experiment 1
    expList = []
    for i in np.arange(0,EXP_1_MAX_PATIENTS,EXP_1_PATIENT_STEP):





    return expList
    

# Also need a function to generate queries
# The function should use high-level parameters

# args = (nPatients, nDims, nSparse, nHeter, 
#        HeterMeans, nIters, nTest, BIC={true,false})
# Return = (AverageMeanDiff, Classification Confusion Matrix, 
#           Histogram/clusters if desired)
def experiment(args):
    nPatients = args[0]
    nDims = args[1]
    nSparse = args[2]
    nHeter = args[3]
    heterMeans = args[4] #delta
    nIters = args[5]
    nTest = args[6]
    Bayes = args[7]

    datgen = dc.DataSim(nCovariates=nDims,
                nHeterogenous=nHeter,
                delta=heterMeans) #needs sparsity
	datmodel = mdl.simModel(generator = datgen, BIC = Bayes)

    errors = []
    accuracies = []
    hists = []
    for i in np.arange(0,nIters):
        # Hopefully different distribution every time
        Xtrain, Ztrain, Ytrain = datmodel.generate(nPatients)
        Xtest, Ztest, Ytest = datmodel.generate(numTest) #Should be same distribution
        # Set sparsity here
        # Can do this later

        datmodel.train(Xtrain, Ztrain, Ytrain)
        accuracy, h1, h2 = datmodel.test(Xtest,Ztest,Ytest)
        if i == 0:
           hists.append(h1)
           hists.append(h2)
           hists.append(datmodel.gmm) # To preserve the means
        errs,avgerrs,stderrs = datmodel.getMeansError()
        errors.append(avgerrs)
        accuracies.append(accuracy)
        # maybe get representative histogram, too?
        datmodel.regenerate() #re-generate the coefficients
    # Next, need to gen mean and stdev error, and min/max/quartiles
    errorArray = np.array(errors)
    accArray = np.array(accuracies)
    results1 = [np.mean(errorArray)),
                np.std(errorArray), 
                np.amin(errorArray),
                np.amax(errorArray),
                np.percentile(errorArray,25),
                np.percentile(errorArray,50),
                np.percentile(errorArray,75)
                ]
    results2 = [np.mean(accArray)),
                np.std(accArray), 
                np.amin(accArray),
                np.amax(accArray),
                np.percentile(accArray,25),
                np.percentile(accArray,50),
                np.percentile(accArray,75)
                ]
    
    # Get same for accuracies
    # Possibly get a representative histogram

    # Return all this data
    return [results1, results2, hists]


def organizeResults(outcomeList):


# Need to implement sparsity
# Need to check to make sure magnitudes of coefficients are not too large or something

# Need to write a function to collect all queries into data structures,
# and then plot them in 1d graphs with error bars, or 2d graphs

# Then, main function generates queries, paritions them, and collects the results
# and sends the collected results (in any order) to the organize/visualization function

