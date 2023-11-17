#####################################################################################################
##### SCRIPT PARA GENERAR DATOS MULTICLASES SIMULADOS Y PROBAR EL ALGORITMO EN VERSION
#### MULTICLASE CON LA ELIMINACION DEL PARAMETRO
#####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Normal_dataset_generator import normal_generator3


seed0 = 1
seed1 = 2
seed2 = 3
n0 = 1000
n1 = 1000
n2 = 1000

## Dataset multiclass 1
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [3, 3]
sigma1 = [[1, 0], [0, 1]]
mu2 = [2, -1]
sigma2 = [[3, 1], [1, 1]]

X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)

