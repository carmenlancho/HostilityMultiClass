#### Unir R y Python
import pandas as pd
import rpy2
import rpy2.robjects as robjects

from rpy2.robjects.packages import importr, data
from rpy2.robjects import numpy2ri
numpy2ri.activate()

from Normal_dataset_generator import *
from Hostility_multiclass_algorithm import hostility_measure_multiclass
import copy
from sklearn.cluster import KMeans



utils = importr('utils')
base = importr('base')

wskm = importr('wskm')
wskm.ewkm

from sklearn import datasets

iris = datasets.load_iris().data


myewkm = wskm.ewkm(iris, 3,**{'lambda':0.5}, maxiter=100, maxrestart = -1)
#If maxrestart< 0 then there is no limit on the number of restarts and we are much more likely to get a full set of k clusters
myewkm.names
myewkm[-1] # weights
aa = pd.DataFrame(myewkm[-1])
aa.sum(axis=1)



###########################################################################################
######              HOSTILITY MEASURE CON EWKM (en lugar de kmeans)                  ######
###########################################################################################

#Example


# Parameters
# seed1 = 1
# seed2 = 2
# n0 = 3000
# n1 = 3000
#
# # Dataset 1
# mu0 = [0, 0]
# sigma0 = [[1, 0], [0, 1]]
# mu1 = [3, 3]
# sigma1 = [[1, 0], [0, 1]]
#
# X, y = normal_generator2(mu0, sigma0, n0, mu1, sigma1, n1, seed1, seed2)
#


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


sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto = hostility_measure_multiclass(sigma, X, y, k_min, seed=0)




def hostility_measure_multiclass_ewkm(sigma, X, y, k_min, seed=0):
    """
    :param sigma: proportion of grouped points per cluster. This parameter automatically determines the number of clusters k in every layer.
    :param X: instances
    :param y: labels
    :param k_min: the minimum number of clusters allowed (stopping condition)
    :param seed: for the k-means algorithm
    :return: host_instance_by_layer - df with hostility instance values per layer (cols are number of clusters per layer, rows are points)
             data_clusters - original data and the cluster to which every original point belongs to at any layer
             results - dataframe (rows are number of clusters per layer) with hostility per class, per dataset and overlap per class
             results_per_class - Pairwise hostility per classes. Rows: who is receiving hostility, columns: who is causing the hostility (proportion of points (row) receiving hostility from the class in the column)
             probs_per_layer - dominance probability of each class in the neighborhood of each point for each layer
             k_auto - automatic recommended value of clusters k for selecting the best layer to stop
    """
    # host_instance_by_layer_df: hostility of each instance in each layer
    # data_clusters: original points and the cluster where they belong to at each layer
    # results:
    # results_per_class:
    # probs_per_layer:
    # k_auto:



    np.random.seed(seed)

    n = len(X)
    n_classes = len(np.unique(y))
    X_aux = copy.deepcopy(X)

    host_instance_by_layer = []

    # first k:
    k = int(n / sigma)
    # The minimum k is the number of classes
    minimo_k = max(n_classes, k_min)
    if k < minimo_k:
        raise ValueError("sigma too low, choose a higher value")
    else:  # total list of k values
        k_list = [k]
        while (int(k / sigma) > minimo_k):
            k = int(k / sigma)
            k_list.append(k)

        # list of classes
        list_classes = list(np.unique(y))  # to save results with the name of the class
        list_classes_total = list(np.unique(y))  # for later saving results
        list_classes_total.append('Total') # for later saving results
        name3 = 'Host_'
        col3 = []
        for t in range(n_classes):
            col3.append(name3 + str(list_classes[t]))

        columns_v =list(col3) + list(['Dataset_Host'])

        # Results is a dataset to save hostility per class, hostility of the dataset and overlap per class in every layer
        index = k_list
        results = pd.DataFrame(0, columns=columns_v, index=index)
        results_per_class = {}
        probs_per_layer = {}

        data_clusters = pd.DataFrame(X)  # to save to which cluster every original point belongs to at any layer
        # prob_bomb = np.zeros(len(X))  # to save the probability, for every original point, of its class in its cluster
        df_bomb = pd.DataFrame(0,columns=list_classes, index=data_clusters.index)

        # dict to save feature weights according to ewkm
        dict_f_w = {}

        h = 1  # to identify the layer
        # k =69
        for k in k_list:

            # kmeds_kmeans = KMeans(n_clusters=k, n_init=15, random_state=seed).fit(X_aux) # kmeans
            np.random.seed(seed)
            kmeds = wskm.ewkm(X_aux, k, **{'lambda': 0.5}, maxiter=1000, maxrestart=-1) # ewkm
            # kmeds.names
            # kmeds[0] # cluster labels
            # kmeds[-1] # variable weights
            # labels_bomb1 = kmeds_kmeans.labels_ # kmeans
            # R starts in 1, not in 0
            labels_bomb1 = kmeds[0] - 1 # ewkm
            f_weights_bomb1 = kmeds[-1] # variable weights

            col_now = 'cluster_' + str(h) # for the data_clusters dataframe
            dict_f_w[col_now] = f_weights_bomb1

            if len(y) == len(labels_bomb1):  # only first k-means
                data_clusters[col_now] = labels_bomb1
                len(labels_bomb1)
                # Probability of being correctly identified derived from first k-means
                table_percen = pd.crosstab(y, labels_bomb1, normalize='columns')
                table_percen_df = pd.DataFrame(table_percen)

                prob_bomb1 = np.zeros(len(X))
                df_bomb1 = pd.DataFrame(columns = list_classes, index = data_clusters.index)
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (labels_bomb1 == i))] = table_percen_df.loc[t, i]
                        df_bomb1[(labels_bomb1 == i)] = table_percen_df.loc[:, i]

            else:  # all except first k-means (which points are in new clusters)
                data2 = pd.DataFrame(X_aux)
                data2[col_now] = labels_bomb1
                data_clusters[col_now] = np.zeros(n)

                for j in range(k):
                    values_together = data2.index[data2[col_now] == j].tolist()
                    data_clusters.loc[data_clusters[col_old].isin(values_together), col_now] = j

                # Proportion of each class in each cluster of the current partition
                table_percen = pd.crosstab(y, data_clusters[col_now], normalize='columns')
                table_percen_df = pd.DataFrame(table_percen)
                prob_bomb1 = np.zeros(len(X))
                df_bomb1 = pd.DataFrame(columns=list_classes, index=data_clusters.index)
                for i in np.unique(labels_bomb1):
                    for t in list_classes:
                        prob_bomb1[((y == t) & (data_clusters[col_now] == i))] = table_percen_df.loc[t, i]
                        df_bomb1[(data_clusters[col_now] == i)] = table_percen_df.loc[:, i]


            # For all cases
            df_bomb += df_bomb1
            # Mean of the probabilities
            df_prob_bomb_mean = df_bomb / h
            prob_self_perspective = np.zeros(len(X))
            for t in list_classes:
                prob_self_perspective[y == t] = df_prob_bomb_mean.loc[y == t, t]

            # We save the dominance probability of every point in every layer
            probs_per_layer[k] = df_prob_bomb_mean


            h += 1  # to count current layer
            col_old = col_now

            #### Data preparation for next iterations
            # New points: medoids of previous partition
            # X_aux = kmeds_kmeans.cluster_centers_ # kmeans
            X_aux = kmeds[1] # centers with ewkm

            ## Hostility instance values in current layer
            host_instance = 1 - prob_self_perspective


            # We binarize putting 1 to the maximum value
            df_binary = pd.DataFrame((df_prob_bomb_mean.T.values == np.amax(df_prob_bomb_mean.values, 1)).T * 1,
                         columns=df_prob_bomb_mean.columns)
            # If there is a tie: no one wins
            df_binary.loc[df_binary.sum(axis=1)>1] = 0
            df_hostility = pd.DataFrame(-1,columns=list_classes, index=data_clusters.index)

            df_classes = pd.DataFrame(columns=list_classes_total, index=list_classes)
            host_vector_binary = np.zeros(n)
            for t in list_classes:
                # If you are the dominant class in your environment
                dominant_condition = (df_binary.loc[:, t] == 1)
                df_hostility.loc[(y==t) & (dominant_condition), t] = 0 # you do not receive hostility from your neighborhood
                # else, you receive hostility
                df_hostility.loc[(y==t) & (~dominant_condition), t] = 1
                host_vector_binary[(y==t)] = df_hostility.loc[(y==t), t]
                # Who is giving hostility? Those classes with more (or equal) presence than you in your environment
                comparison_higher_presence = (df_prob_bomb_mean.loc[(y == t), (df_prob_bomb_mean.columns != t)].values >= df_prob_bomb_mean.loc[
                    (y == t), (df_prob_bomb_mean.columns == t)].values) * 1
                df_hostility.loc[(y==t), (df_hostility.columns != t)] = comparison_higher_presence

                total_hostility_class_t = df_hostility.loc[y == t, t].mean(axis=0)
                hostility_received_per_class = np.array(df_hostility.loc[y == t, (df_hostility.columns != t)].mean(axis=0))

                df_classes.loc[df_classes.index == t,df_classes.columns == t] = 0
                df_classes.loc[df_classes.index == t, df_classes.columns == 'Total'] = total_hostility_class_t
                df_classes.loc[df_classes.index == t,
                (df_classes.columns != t) & (df_classes.columns != 'Total')] = hostility_received_per_class
            # We save detail of pairwise hostility relation in each layer
            results_per_class[k] = df_classes
            host_dataset = np.mean(host_vector_binary) # hostility of the dataset


            # Save results from all layers
            host_instance_by_layer.append(host_instance)
            results.loc[k] = df_classes['Total'].tolist() + [host_dataset]


        ## Automatic selection of layer
        results_aux = results.loc[:, results.columns.str.startswith('Host')]
        change_max = results_aux.iloc[0, :] * 1.25
        change_min = results_aux.iloc[0, :] * 0.75
        matching = results_aux[(results_aux <= change_max) & (results_aux >= change_min)]
        matching.dropna(inplace=True)  # values not matching appear with NaN, they are eliminated
        k_auto = matching.index[-1] # k value from last layer matching the condition of variability

    host_instance_by_layer = np.vstack(host_instance_by_layer)
    host_instance_by_layer_df = pd.DataFrame(host_instance_by_layer.T, columns=results.index)

    return host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto, dict_f_w



sigma = 5
delta = 0.5
seed = 0
k_min = 0
host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto, dict_f_w = hostility_measure_multiclass_ewkm(sigma, X, y, k_min, seed=0)


import random
import math

a = math.floor(min(min(X[0]),min(X[1])))
b = math.ceil(max(max(X[0]),max(X[1])))
random_float_list = []
for i in range(len(y)):
    x = round(random.uniform(a, b), 2)
    random_float_list.append(x)

var_red = np.array(random_float_list)
X_df = pd.DataFrame(X,columns=['a','b'])
X_df['c'] = var_red
X_redundant = X_df.to_numpy()

host_instance_by_layer_df, data_clusters, results, results_per_class, probs_per_layer, k_auto, dict_f_w = hostility_measure_multiclass_ewkm(sigma, X_redundant, y, k_min, seed=0)










def feature_weight_Host(dict_f_w):

    layer = 0
    for l in dict_f_w.keys():
        layer+=1
        print(dict_f_w[l])
        df_layer = pd.DataFrame(dict_f_w[l])
        df_layer.boxplot(return_type='axes',meanline=True, showmeans=True);
        title_boxplot = 'Layer ' +str(layer)
        plt.title(title_boxplot)
        plt.show()
        df_layer.describe()

    return
# https://stackoverflow.com/questions/51087522/combine-histogram-with-describe-in-python

# from sklearn import preprocessing
# import os
#
# root_path = os.getcwd()
#
# path_csv = os.chdir(root_path+'/HostilityMultiClass/datasets')
# df = pd.read_csv('bupa.csv')
# X = df.iloc[:,:-1].to_numpy() # all variables except y
# X = preprocessing.scale(X)
# y = df[df.columns[-1]].to_numpy()
#
