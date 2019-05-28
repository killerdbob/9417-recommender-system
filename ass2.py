import pandas as pd
import numpy as np
from numpy import linalg as la
from random import random,seed
from scipy.sparse.linalg import svds
import scipy.sparse as sp
import time

def dimension_reduction(mat,percentage):
    user,weight,item = svds(mat)
    weight /= max(weight)
    ## square and resort desending
    weight2 = weight**2
    re_order = np.argsort(-weight2)
    weight2 = weight2[re_order]
    user = user.T[re_order].T
    
    ## only use large weight
    ## depend on percentage
    t = np.nansum(weight2)*percentage
##    print("t=",t)
    s = 0
    for i in range(len(weight2)):
        s += weight2[i]
        if s >= t:
            reduce_dimension = user.copy()
            reduce_dimension.resize((len(user),i+1))
            return reduce_dimension


## cos sim
## rescale to 0-1
def cosSim(inA,inB):
    num=np.dot(inA,inB)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

#data_mat: training matrix
#weight_mat: svd weighted matrix
#numbers of k nearest user
def estimate_rating(data_mat,weight_mat,userid,movieid,k=10):
    ind = data_mat.T[movieid].nonzero()[1]
    sim = np.zeros(len(ind))
    j=0
    for i in ind:
        if i==userid:
            sim[j]=-1
        else:
            sim[j]=cosSim(weight_mat[i],weight_mat[userid])
        j+=1
    order = np.argsort(-sim)[:k]

    sum_sim = 0
    sum_rate = 0
    for i in order:
        sum_sim += sim[i]
        # print(data_mat[ind[i],movieid],"         ",ind[i],"         ",movieid)
        sum_rate += data_mat[ind[i],movieid]*sim[i]
    # print(sum_rate/sum_sim)
    if sum_sim == 0:return 0
    return sum_rate/sum_sim
        

if __name__ == '__main__':
    print("-----------------------Start--------------------------")
    print("--------------------Please Wait ----------------------")
    ## Load Data 
    ## Get user-movie ratings from MovieLens('ratings.csv')
    #####################################read in data start############################################
    dataFile = 'ratings2.csv'
    Col_Names = ['userId', 'movieId', 'rating', 'timestamp']
    data_chunks = pd.read_csv(dataFile,
                              dtype = {'userId':np.int32,'movieId':np.int32,'rating':np.float,'timestamp':np.int64},
                              skiprows=1, names=Col_Names,
                              chunksize=500000)

    # data[Col_Names[:-1]] = data[Col_Names[:-1]].apply(pd.to_numeric)
    #######################################read in data end############################################
    time.clock()
    # print(data)
    ## Get user amount and item amount
    user_num = 0
    item_num = 0
    record_num = 0
    for data in data_chunks:
        user_num = max(max(data.userId),user_num)
        item_num = max(max(data.movieId),item_num)
        record_num += data.T.shape[1]

    print("user number: "+str(user_num)+"\nitem_num: "+str(item_num))
    print("time: "+str(time.clock()))
    print("-------------------------------------------------")
    data_chunks.close()
    #####################################build up matrix start############################################
    ## Data process: split training and test data set:
    ## We use about 90% training-10% testing cross validation
    t=40/record_num#rates of test samples
    data_chunks = pd.read_csv(dataFile,
                              dtype = {'userId':np.int32,'movieId':np.int32,'rating':np.float,'timestamp':np.int64},
                              skiprows=1, names=Col_Names,
                              chunksize=500000)
    # train_mat = np.zeros((user_num, item_num))#training matrix
    train_mat = sp.lil_matrix((user_num, item_num))
    test_data = []#test dataset

    seed(0)

    train_num = 0
    for data in data_chunks:
        print(".",end="")
        for record in data.itertuples():
            if random()<t or len(test_data)==0:
                test_data.append([record[1]-1, record[2]-1, record[3]])
            else:
                train_mat[record[1]-1, record[2]-1] += record[3]
                train_num += 1
    print()
    print("time: " + str(time.clock()))
    data_chunks.close()
    #######################################build up matrix end############################################

    print("number of test data: "+str(len(test_data))+"\nnumber of records: "+str(record_num))
    #####################################base line start############################################
    print("-------------------------Base line predictor------------------------\n")
    bias=0
    users_bias = []
    movies_bias = []
    # getting mu of matrix
    num=0
    total=0
    previous_time = time.clock()

    user_sum = train_mat.sum(1).T.tolist()[0]
    num_user = [ d.count_nonzero() for d in train_mat ]
    bias = sum(user_sum)/sum(num_user)
    # print(sum(user_sum)/sum(num_user))
    for i in range(train_mat.T.shape[1]):
        if(num_user[i]!=0):
            users_bias.append(user_sum[i]/num_user[i]-bias)
        else:
            users_bias.append(0)

    movies_sum = train_mat.T.sum(1).T.tolist()[0]
    num_movies = [ sum([1 if i>0 else 0 for i in d]) for d in train_mat.T.data ]
    # print(len(movies_sum))
    for i in range(train_mat.shape[1]):
        if(num_movies[i]!=0):
            movies_bias.append(movies_sum[i]/num_movies[i]-bias)
        else:
            movies_bias.append(0)

    square_error = 0
    for test in test_data:
        uid=test[0]#user id
        mid=test[1]#movie id
        td=test[2]#test data
        try:
            square_error += (bias+users_bias[uid]+movies_bias[mid]-td)**2
        except:
            print(uid,mid)
    square_error /= len(test_data)
    print("square_error = "+str(square_error)+"\n")
    print("time: " + str(time.clock()-previous_time))
    print("------------------------Base line predictor END-----------------------\n")
    #######################################base line end############################################


    #####################################collaborative filtering start############################################
    for k in [10,20,50,100]:
        print("-----------------------Collaborative filtering-k := "+str(k)+"-------------------------")
        for percentage in [0.95,0.9,0.8,0.5,0.3,0.1][::-1]:
            user = dimension_reduction(train_mat,percentage)
            square_error = 0
            # print("--------")
            # print(len(test_data))
            previous_time = time.clock()
            for test in test_data:
                square_error += (estimate_rating(train_mat,user,test[0],test[1])-test[2])**2
                # print(square_error)
            square_error /= len(test_data)
            print("k(similar users) = ",k,"\nsvd percentage = "+str(percentage)+"\nsquare_error = "+str(square_error))
            print("time: " + str(time.clock()-previous_time))
            print("-------------------------------------------------")
    print("-----------------------Collaborative filtering END-----------------------")
    #######################################collaborative filtering end############################################



