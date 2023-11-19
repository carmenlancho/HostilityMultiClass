#####################################################################################################
##### SCRIPT PARA GENERAR DATOS MULTICLASES SIMULADOS Y PROBAR EL ALGORITMO EN VERSION
#### MULTICLASE CON LA ELIMINACION DEL PARAMETRO
#####################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Normal_dataset_generator import normal_generator3, normal_generator4, normal_generator5
from Hostility_multiclass_algorithm import hostility_measure_multiclass


seed0 = 1
seed1 = 2
seed2 = 3
seed3 = 4
seed4 = 5
n0 = 1000
n1 = 1000
n2 = 1000
n3 = 1000
n4 = 1000


## Dataset multiclass 1 (3 separadas)
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [6, 5]
sigma1 = [[1, 0], [0, 1]]
mu2 = [6, -5]
sigma2 = [[1, 0], [0, 1]]

X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)

sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)



## Dataset multiclass 2
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [3, 3]
sigma1 = [[1, 0], [0, 1]]
mu2 = [2, -1]
sigma2 = [[3, 1], [1, 1]]

X, y = normal_generator3(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, seed0, seed1, seed2)


sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)



## Dataset multiclass 3
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [8, 7]
sigma1 = [[1, 0], [0, 1]]
mu2 = [1, -3]
sigma2 = [[0.3, 0], [0, 5]]
mu3 = [7, 3]
sigma3 = [[1, 0], [0, 1]]

X, y = normal_generator4(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, mu3, sigma3, n3, seed0, seed1, seed2, seed3)



sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)


## hacer un ejemplo con una normal dentro de otra totalmente
# una clase que solape con el resto
# otro con imbalanced




## Dataset multiclass 4
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [8, 7]
sigma1 = [[1, 0], [0, 1]]
mu2 = [3, -2]
sigma2 = [[0.3, 0], [0, 5]]
mu3 = [8, 7]
sigma3 = [[0.1, 0], [0, 0.1]]
mu4 = [4, 4]
sigma4 = [[7, 0], [0, 8]]

X, y = normal_generator5(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, mu3, sigma3, n3, mu4, sigma4, n4, seed0, seed1, seed2, seed3, seed4)



sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)





## Dataset multiclass 5
mu0 = [0, 0]
sigma0 = [[1, 0], [0, 1]]
mu1 = [8, 7]
sigma1 = [[1, 0], [0, 1]]
mu2 = [3, -2]
sigma2 = [[0.3, 0], [0, 5]]
mu3 = [8, 7]
sigma3 = [[0.1, 0], [0, 0.1]]
mu4 = [4, 4]
sigma4 = [[7, 0], [0, 8]]

n1 = 100
n2 = 50

X, y = normal_generator5(mu0, sigma0, n0, mu1, sigma1, n1, mu2, sigma2, n2, mu3, sigma3, n3, mu4, sigma4, n4, seed0, seed1, seed2, seed3, seed4)



sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)



