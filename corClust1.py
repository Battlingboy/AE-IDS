from sklearn.cluster import AffinityPropagation
import numpy as np
import time
#class corClust1:

def cluster(A):
    start = time.time()
    A = A.T
    A = np.mean(A,axis=1)
    clustering = AffinityPropagation(max_iter=1000).fit(A.reshape(-1,1))#max_iter=300
    labels = list(clustering.labels_)
    size1 = list(set(labels))
    Map = []
    Value = []
    for size in size1:
        index = []
        value = []
        for i in range(len(labels)):
            if labels[i] == size:
                index.append(i)
                value.append(A[i])
        Map.append(index)
        Value.append(value)
    stop = time.time()
    print("CompleteTime: "+ str(stop - start))
    print(Map)
    for v in Value:
        print(v)
    return Map

    
   