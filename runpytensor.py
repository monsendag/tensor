import logging
import numpy
from numpy import genfromtxt

import sys
sys.path.insert(0, './pytensor')

import sptensor

# Set logging to DEBUG to see CP-ALS information
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

file = '../datasets/movielens-synthesized/ratings-synthesized-50k.csv'
logging.debug("Loading dataset from file: %s", file)
data = genfromtxt(file, delimiter=',', skip_header=1)

logging.debug("Loaded data")

# we need to convert data into two lists; subscripts/coordinates and values
n = len(data)

subs_1 = numpy.append(data[:,:2], numpy.zeros((n, 1)), 1)
subs_2 = numpy.append(data[:,:2], numpy.ones((n, 1)), 1)

subs = numpy.vstack([subs_1, subs_2])
subs = subs.astype(int)

vals = numpy.hstack([data[:,2], data[:, 3]])
vals = vals.flatten()
vals = [[x] for i,x in enumerate(vals)]
vals = numpy.array(vals)


spten2 = sptensor.sptensor(subs, vals)
print spten2.shape