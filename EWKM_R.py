#### Unir R y Python
import pandas as pd
import rpy2
import rpy2.robjects as robjects

from rpy2.robjects.packages import importr, data
from rpy2.robjects import numpy2ri
numpy2ri.activate()

utils = importr('utils')
base = importr('base')

wskm = importr('wskm')
wskm.ewkm

from sklearn import datasets

iris = datasets.load_iris().data


myewkm = wskm.ewkm(iris, 3,**{'lambda':0.5}, maxiter=100)
myewkm.names
myewkm[-1] # weights