import logging
import numpy
from numpy import genfromtxt
from sktensor import sptensor, cp_als
from sktensor.rescal import als as rescal_als
from scipy.sparse import lil_matrix
from scipy.io.matlab import loadmat

# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

file = '../datasets/movielens-synthesized/ratings-synthesized-50k.csv'
logging.debug("Loading dataset from file: %s", file)
data = genfromtxt(file, delimiter=',')

logging.debug("Loaded data")

# we need to convert data into two lists; subscripts/coordinates and values
n = len(data)


subs_1 = numpy.append(data[:,:2], numpy.zeros((n, 1)), 1)
subs_2 = numpy.append(data[:,:2], numpy.ones((n, 1)), 1)

subs = numpy.vstack([subs_1, subs_2])
subs = subs.astype(int)

vals = numpy.hstack([data[:,2], data[:, 3]])
vals = vals.flatten()

# convert subs tuple of arrays (rows, cols, tubes)
subs = (subs[:,0], subs[:,1], subs[:,2])

# load into sparse tensor
T = sptensor(subs, vals)

logging.debug("Starting Tucker decomposition")

T = loadmat('../datasets/alyawarra/alyawarradata.mat')['Rs']
X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

X = [lil_matrix(T[:, :, k]) for k in range(T.shape[2])]

# Decompose tensor using RESCAL-ALS
A, R, fit, itr, exectimes = rescal_als(X, 100, init='nvecs', lambda_A=10, lambda_R=10)
#P, fit, itr, exectimes = tucker_hooi(T, [10, 10, 2], init='random')
logging.debug("Finished tucker decomposition")

#P = core.ttm(U)

#core

#rating = core[1,1193,0] * (U[0][1,:] * U[1][1193,:] * U[2][0 , :])

#print P[1,1193,0] # 5
#print P[1,661, 0] # 3
#print P[1,594, 1] # 1.6
#print P[1,1193, 1] # 2.2
