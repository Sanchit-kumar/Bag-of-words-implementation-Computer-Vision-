import numpy as np
import cv2
from scipy.spatial import distance
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

n_test=0
n_train=0
x_train=[]
y_train=[]
x_test=[]
y_test=[]
clusters_center=[]
train_bovw_feature_dict={}
test_bovw_feature_dict={}
all_visual_feature_list=[]

K_CLUSTER=105 ## Value of k I found by using k_cluster_value_selection which make graph between inertia vs K

def readingData():
    global n_test,n_train,x_train,y_train,x_test,y_test
    train=pd.read_csv("fashion-mnist_train.csv") #reading data from csv file
    test=pd.read_csv("fashion-mnist_test.csv") 
    
    train_y_df=train[train.columns[0]]
    train_x_df=train[train.columns[1:]]
    test_y_df=test[test.columns[0]]
    test_x_df=test[test.columns[1:]]

    n_test=test_x_df.shape[0]
    n_train=train_x_df.shape[0]

    # n_test=1000   ##For quick results on small dataset, please uncomment only these 2 likes and comment above 2 lines. It will work.
    # n_train=1000  ## However, small dataset will compromise the performace and accuracy.

    for i in range(n_test): #making train data and labels list
        x_test.append(test_x_df.iloc[i].values.reshape(28,28).astype(np.uint8))
        y_test.append(test_y_df.iloc[i])
        
    for i in range(n_train): #making train data and labels list
        x_train.append(train_x_df.iloc[i].values.reshape(28,28).astype(np.uint8))
        y_train.append(train_y_df.iloc[i])

def make_dictionary(x,y,n): ###HELPER FUNCTION (MAKE DICTIONARY OF THE IMAGES OF SAME CATEGORY)
    images={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    for i in range(n):
        images[y[i]].append(x[i])
    return images

def feature_extractor(bovw_image_dict): ## It will extract the key point features from the images and group them
                                        ## and return the all features from all images and dictionary of features w.r.t category
    visual_feature_list = []
    visual_feature_dict = {}
    sift = cv2.xfeatures2d.SIFT_create()
    for key,images in bovw_image_dict.items():
        extracted_features_list = []
        for img in images:
            keypoints, descriptor = sift.detectAndCompute(img,None)
            if len(keypoints)<1:
                continue
            visual_feature_list.extend(descriptor)
            extracted_features_list.append(descriptor)
        visual_feature_dict[key] = extracted_features_list
    return [visual_feature_list, visual_feature_dict]


def kmeans(feature_descriptor,K=105): ## I AM USING K-MEANS ALGORITHM FOR MAKING THE CLUSTERS
    model = KMeans(n_clusters = K, n_init=20, init='k-means++') ##Upto 20 iterations
    model.fit(feature_descriptor)
    visual_words_centers = model.cluster_centers_ 
    return [visual_words_centers,model]

def kmeans_inertia(feature_descriptor,K): ##FINDING THE INERTIA, SO THAT WE CAN COMPARE WITH OTHER K VALUES TO FIND BEST K VALUE
    model = KMeans(n_clusters = K, n_init=20, init='k-means++')
    model.fit(feature_descriptor)
    return model.inertia_

def k_cluster_value_selection(feature_list): #it is used to plot the graph between inertial and k (#of clusters) to find the 
                                            # best value of the k

    inertia=[]
    values= [10*i for i in range(1,16)]
    for k in values:
        temp=kmeans_inertia(feature_list,k)
        inertia.append(temp)
    plt.figure(figsize=(12,8))
    plt.plot(values,inertia,marker='o')
    plt.ylabel('Inertia')
    plt.xlabel('Number of clusters')
    plt.show()
#     plt.savefig("K-value-selection.jpg")



def CreateDictionary(): ################# CREATING THE DICTIONARY ##############
    readingData()
    global clusters_center,train_bovw_feature_dict,test_bovw_feature_dict,all_visual_feature_list
        
    train_dict=make_dictionary(x_train,y_train,n_train) #it will create the dictionary of the category and images
    test_dict=make_dictionary(x_test,y_test,n_test)
    
#     np.save("Visual_dict_train.npy",train_dict) #saving the visual dictionary
#     np.save("Visual_dict_test.npy",test_dict)
    

    extracted_features= feature_extractor(train_dict)
    all_visual_feature_list = extracted_features[0]
    train_bovw_feature_dict = extracted_features[1] # bag of word of extracted features dictionary

#     np.save("Visual_dict_extracted_features.npy",train_bovw_feature_dict)
    
    test_bovw_feature_dict = feature_extractor(test_dict)[1]

    [clusters_center,kmodel]= kmeans(all_visual_feature_list,K_CLUSTER)

    
    ### SAVING MOST CLOSEST VISUAL WORDS TO THE MEAN OF EACH CLUSTERS ###
    
    closest_indexs, _ = pairwise_distances_argmin_min(clusters_center,all_visual_feature_list) 
    closest_visual_word={}
    for i in range(len(closest_indexs)):
        closest_visual_word[i]=all_visual_feature_list[closest_indexs[i]]

    np.save("closest_visual_word_dictionary_with_class.npy",closest_visual_word) #saving "dictionary class(key) to closest feature (value)"
    np.save("clusters_center_points.npy",clusters_center) #K-class clusters center points from which closest feature in obtained"
    




def ComputeHistogram(visual_dict_mat,clusters_center): #Assuming feature ventor is the center pointes of each cluster
        
        dict_feature = {}
        k=len(clusters_center)
        model= KMeans(n_clusters=k, init=clusters_center, max_iter=1) # loading kmeans model with previously features as cluster center points 
        model.fit(clusters_center) #using the same features as the center of teh clusters.
                                    #So, kmean algorithm internally allot the unknown feature (to be predicted) to the nearest neighbour
                                    # and hence giving the weight to the nearest neighbour
        n=len(visual_dict_mat)
        for i in range(n):
            category = []
            for img in visual_dict_mat[i]:
                histogram = np.zeros(k) #INITIAL EMPTY HISTOGRAM, THE VALUES WILL BE ALLOCATED USING MODEL PREDICTION
                                        # MODEL PREDICTION is not learning anything, but considering given centers of the cluster
                                        # and predicting the class on the basis of its closeness in all clusters
                for each_feature in img:
                    label_class=model.predict([each_feature])[0]
                    histogram[label_class] += 1
                category.append(histogram)
            dict_feature[i] = category
        return dict_feature


def MatchHistogram(hist1,hist2): #################
    return 0.5*np.sum((hist1-hist2)**2/(hist1+hist2+1e-6)) ##chi square distance between two histogram
#     return distance.euclidean(hist1,hist2) ### Euclidean distance between two histogram

def KNN(histogram_bovw_train, histogram_bovw_test):
    
    confusion_matrix={}
    for i in range(10):
        confusion_matrix[i] = [0,0,0,0] #true +ve, total , false +ve,false -ve (true -ve not required, instead I took total)
    total_test = 0
    correct_predict = 0
    
    for tst_key, tst_value in histogram_bovw_test.items():
        # confusion_matrix[tst_key] = [0,0,0,0] #true +ve, total , false +ve,false -ve (true -ve not required, instead I took total)
        for tst in tst_value:
            predicted_key =0
            predict_start = 0
            minimum_distance = 0
            for train_key,train_value in histogram_bovw_train.items():
                for train in train_value:
                    if(predict_start == 0):
                        minimum_distance = MatchHistogram(tst,train)  #calculating the euclidean distance
                        predicted_key = train_key
                        predict_start += 1
                    else:
                        dist = MatchHistogram(tst,train)
                        if(dist < minimum_distance):
                            minimum_distance = dist #updating the minimum eucledian distance
                            predicted_key = train_key
            
            if(tst_key ==predicted_key):
                confusion_matrix[tst_key][0] += 1
                correct_predict += 1
            else:
                confusion_matrix[predicted_key][2]+=1 #updating false +ve
                confusion_matrix[tst_key][3]+=1 #updatign false -ve
            confusion_matrix[tst_key][1] += 1
            total_test += 1

    return [total_test, correct_predict, confusion_matrix] #this confusion matrix is customized cunfusion matrix

def performance_matrix(result_matrix):
    avg_acc = (result_matrix[1] / result_matrix[0]) * 100
    print("Average accuracy: %" + str(avg_acc))
    print("\nClass based performance: \n")
    for key,value in result_matrix[2].items():
        if value[1]==0:
            value[1]=1
        accuracy = (value[0] / value[1]) * 100
        precision=(value[0]/(value[0]+value[2]))
        recall=(value[0]/(value[0]+value[3]))
        print(key,"  Accuracy= %",str(accuracy), "Precision=",str(precision), "Recall:",str(recall))
