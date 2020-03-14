# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:09:11 2019
180+100+100+30+
@author: Mr.L
0"""
import KitNET as kit
import numpy as np
import pandas as pd
import time
X = pd.read_csv(r'C:\Users\Mr.L\Desktop\KitNET-py-master\Processed Traffic Data for ML Algorithms\DDOS attack-HOIC.csv',low_memory=False)
l = set(X["Label"])
print(l)

A = X[(X["Label"]=='Benign')]
B = X[(X["Label"]=='DDOS attack-HOIC')]

X = pd.concat([A,B],axis=0,ignore_index=True)
print(len(X))
l = set(X["Label"])
print(l)
#print("Sparse Fwd<=1")
#X = X[(X["TotLen Fwd Pkts"]<=1)]#SparseMatrix{Bot}
#print("Sparse Fwd>1")
#X = X[(X["TotLen Fwd Pkts"]>1)]#DenseMatrix{Bot}
##
print("Sparse Bwd==0")
X = X[(X["TotLen Bwd Pkts"]==0)]#SparseMatrix{DDOS:HOIC}
#print("Sparse Bwd!=0")
#X = X[(X["TotLen Bwd Pkts"]!=0)]#DenseMatrix{DDOS:HOIC}
#
#print("Sparse Fwd==0")
#X = X[(X["TotLen Fwd Pkts"]==0)]#SparseMatrix{DDOS:LOIC-UDP}0.2
#print("Sparse Fwd!=0")
#X = X[(X["TotLen Fwd Pkts"]!=0)]#SparseMatrix{DoS:LOIC-UDP}

#print("Sparse Fwd==0")
#X = X[(X["TotLen Fwd Pkts"]==0)]#SparseMatrix{Brute Force -XSS}0.4
#print("Sparse Fwd!=0")
#X = X[(X["TotLen Fwd Pkts"]!=0)]#SparseMatrix{Brute Force -XSS}0.4

#print("Sparse Fwd=0")
#X = X[(X["TotLen Fwd Pkts"]==0)]#SparseMatrix{Brute Force -Web}0.4
#print("Sparse Fwd!=0")
#X = X[(X["TotLen Fwd Pkts"]!=0)]#SparseMatrix{Brute Force -Web}0.4

#print("Sparse Bwd==0")
#X = X[(X["TotLen Bwd Pkts"]==0)]#SparseMatrix{SQL Injection}
#print("Sparse Bwd!=0")
#X = X[(X["TotLen Bwd Pkts"]!=0)]#DenseMatrix{SQL Injection}

#print("Sparse Fwd==0")
#X = X[(X["TotLen Fwd Pkts"]==0)]#SparseMatrix{Infilteration}
#print("Sparse Fwd!=0")
#X = X[(X["TotLen Fwd Pkts"]!=0)]#SparseMatrix{Infilteration}

#print("Sparse Bwd==0")
#X = X[(X["TotLen Bwd Pkts"]==0)]#SparseMatrix{FTP-BruteForce}
#print("Sparse Bwd!=0")
#X = X[(X["TotLen Bwd Pkts"]!=0)]#DenseMatrix{FTP-BruteForce}

#print("Sparse Bwd==0")
#X = X[(X["TotLen Bwd Pkts"]==0)]#SparseMatrix{SSH-Bruteforce}
#print("Sparse Bwd!=0")
#X = X[(X["TotLen Bwd Pkts"]!=0)]#DenseMatrix{SSH-Bruteforce}

X = X.replace("Infinity",0)
X = X.replace(np.NaN,0)
X = X.drop(["Timestamp"], axis=1)
X = X.replace("Benign",0)
X = X.replace("DDOS attack-HOIC",1)
X = X.sort_values(["Label"],ascending=True)

Label = np.array(X["Label"])
y_test = Label
X = X.drop(["Label"], axis=1)
X = X.astype(float)
X = X.as_matrix()

from collections import Counter
z = Counter(Label)
print(z)
print(len(X))

#up-sample
#if len(z)!=1:
#    if z[1]<int(np.trunc(len(X)*0.1)): 
#        from imblearn.over_sampling import RandomOverSampler
#        ros = RandomOverSampler(random_state=0)
#        X1, Label1 = ros.fit_sample(X[int(np.trunc(z[0]*0.85)):], Label[int(np.trunc(z[0]*0.85)):])
#        X = np.vstack((X[0:int(np.trunc(z[0]*0.85))],X1))
#        y_test = np.hstack((Label[0:int(np.trunc(z[0]*0.85))],Label1))

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=1000,n_jobs=-1)#,bootstrap=True,oob_score=True)#
clf = clf.fit(X, y_test)#,max_depth=20
#print(clf.oob_score_)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
importanceindex = []
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    
from collections import Counter
z1 = Counter(y_test)
print(z1)



from sklearn.preprocessing import Normalizer
norm1 = Normalizer(norm='l2')
X = norm1.fit_transform(X)

#KitNET params:
Ratio = 0.85
FMgrace = int(np.trunc(z1[0]*Ratio*0.25))#the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = int(np.trunc(z1[0]*Ratio*0.75))#the number of instances used to train the anomaly detector (ensemble itself)
# Build KitNET
K = kit.KitNET(X.shape[1],FMgrace,ADgrace)
RMSEs = np.zeros(X.shape[0]) # a place to save the scores
#S = np.zeros(X.shape[0])
print("Running KitNET:")

start = time.time()
for i in range(X.shape[0]):
    if i % 10000 == 0:
        print(i)
    RMSEs[i] = K.process(X[i,],X) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

X_train = RMSEs[FMgrace+ADgrace+1:].reshape(-1,1)
y_test1 = y_test[FMgrace+ADgrace+1:]

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(X_train)
y_pred = gmm.predict(X_train)

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2)
clf.fit(X_train)
y_pred = clf.labels_


#for i in range(len(y_pred)):
#    if y_pred[i]==0:
#        y_pred[i]=1
#    else:
#        y_pred[i]=0

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test1, y_pred)
print("accuracy_score",acc)
from sklearn.metrics import precision_score
pre = precision_score(y_test1, y_pred)
print("precision_score",pre)
from sklearn.metrics import recall_score
rec = recall_score(y_test1, y_pred)
print("recall_score",rec)
from sklearn.metrics import f1_score
print("f1_score",f1_score(y_test1, y_pred))
#
from sklearn.metrics import roc_curve
FTP,TPR,threshold = roc_curve(y_test1,y_pred)
print("ROC_AUC Score")
import matplotlib.pyplot as plt
plt.plot(FTP,TPR)
plt.show()
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test1,y_pred))

from sklearn.metrics import confusion_matrix
print("confusion matrix")
print(confusion_matrix(y_test1,y_pred))

# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
#from scipy.stats import norm
#benignSample = np.log(RMSEs[FMgrace+ADgrace+1:71000])
#logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))
#
## plot the RMSE anomaly scores
#print("Plotting results")
#from matplotlib import pyplot as plt
#from matplotlib import cm
#plt.figure(figsize=(10,5))
#fig = plt.scatter(timestamps[FMgrace+ADgrace+1:],RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
#plt.yscale("log")
#plt.title("Anomaly Scores from KitNET's Execution Phase")
#plt.ylabel("RMSE (log scaled)")
#plt.xlabel("Time elapsed [min]")
#plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662],RMSEs[71662]), xytext=(timestamps[58000],1),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662],1), xytext=(timestamps[55000],5),arrowprops=dict(facecolor='black', shrink=0.05),)
#figbar=plt.colorbar()
#figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
#plt.show()
