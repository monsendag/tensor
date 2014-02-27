import logging
import numpy
from numpy import genfromtxt
from sktensor import sptensor, cp_als
# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG)

data = genfromtxt('../datasets/movielens-synthesized/ratings-synthesized-50k.csv', delimiter=',')

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

# Decompose tensor using CP-ALS
P, fit, itr, exectimes = cp_als(T, 500, init='random')

P = P.totensor()

print P[1,1193,0] # 5
print P[1,661, 0] # 3
print P[1,594, 1] # 1.6
print P[1,1193, 1] # 2.2



#print numpy.allclose(T, P)
#print P.U[0].shape
#print "-------"
##print P.U[1].shape
#print "-------"
#print P.U[2].shape
