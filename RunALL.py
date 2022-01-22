
##2020AIM1009
##SANCHIT KUMAR
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import one as on

# n_test=on.n_test
# n_train=on.n_train
# x_train=on.x_train
# y_train=on.y_train
# x_test=on.x_test
# y_test=on.y_test
# clusters_center=on.clusters_center
# train_bovw_feature_dict=on.train_bovw_feature_dict
# test_bovw_feature_dict=on.test_bovw_feature_dict
# all_visual_feature_list=on.all_visual_feature_list
# K_CLUSTER=on.K_CLUSTER

on.CreateDictionary()  ## IT WILL READ DATA FROM THE TEST & TRAIN FILES
                    ### IT WILL ALSO SAVE MOST CLOSEST VISUAL WORDS TO THE MEAN OF EACH CLUSTERS ###
                    ## It will extract the key point features from the images and group them
                    ### SAVING MOST CLOSEST VISUAL WORDS TO THE MEAN OF EACH CLUSTERS ###
                    ### IT will crate necessary dictionary and 
                
histogram_bovw_train = on.ComputeHistogram(on.train_bovw_feature_dict, on.clusters_center)  # Creates histograms for train data    

histogram_bovw_test = on.ComputeHistogram(on.test_bovw_feature_dict, on.clusters_center) # Creates histograms for test data

on.k_cluster_value_selection(on.all_visual_feature_list) ##IT will display graph between K-cluster value
                                                        ## and intertia (K-value from 10-150)

result_matrix = on.KNN(histogram_bovw_train,histogram_bovw_test)

# print("ok")
on.performance_matrix(result_matrix)