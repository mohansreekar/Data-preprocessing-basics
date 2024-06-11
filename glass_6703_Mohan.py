# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:20:39 2024

@author: Mohan Sreekar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
import scipy.stats as stats
import pylab
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as skmet
#reading data from dataset
glass=pd.read_csv(r"C:\Users\Mohan Sreekar\Desktop\project\glass.csv")
glass
# exploratory data analysis
means = []
sd=[]
var1=[]
medin=[]
mod=[]
kur=[]
ske=[]
range1=[]
i=0
while(i<10):
    means.append(glass.iloc[i].mean())
    medin.append(glass.iloc[i].median())
    mod.append(glass.iloc[i].mean())
    var1.append(glass.iloc[i].var())
    sd.append(glass.iloc[i].std())
    kur.append(glass.iloc[i].kurt())
    ske.append(glass.iloc[i].skew())
    range1.append(max(glass.iloc[i])-min(glass.iloc[i]))
    i=i+1
#moment 1
means
medin    
mod
#moment 2
sd
var1
range1
#moment 3
ske
#moment 4
kur

#preprocessing the data
#typecasting
glass.dtypes
glass['Ba']=glass['Ba'].astype('int64')
#we can see all data is in float except Type col which is our output
#duplicates
#for knowing duplicate record in data we use
duplicate=glass.duplicated(keep="last")
duplicate
#there is only one duplicate value
glass1=glass.drop_duplicates(keep="last")
glass1
#outlier analysis
#handling outliers using winsorization
winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RI','Na','Mg','Al','Si','K','Ca','Fe'])
glass1=winsor_iqr.fit_transform(glass1)
sns.boxplot(glass1.RI)
sns.boxplot(glass1.Na)
sns.boxplot(glass1.Mg)
sns.boxplot(glass1.Al)
sns.boxplot(glass1.Si)
sns.boxplot(glass1.K)
sns.boxplot(glass1.Ca)
sns.boxplot(glass1.Fe)
winsor_iqr=Winsorizer(capping_method='iqr',tail='both',fold=3,variables=['Ba'])
glass1=winsor_iqr.fit_transform(glass1)
sns.boxplot(glass1.Ba)
#
#missing values 
glass1.isna().sum()
#there are no missing values
#transformation
stats.probplot(glass1.RI,dist="norm", plot=pylab)
stats.probplot(pow(glass1.RI,65),dist="norm",plot=pylab)
glass1.RI=pow(glass1.RI,65)
stats.probplot(glass1.Mg,dist="norm", plot=pylab)
stats.probplot(pow(glass1.Mg,10),dist="norm", plot=pylab)
glass.Mg=pow(glass1.Mg,10)
stats.probplot(glass1.Si,dist="norm", plot=pylab)
stats.probplot(pow(glass1.Si,10),dist="norm", plot=pylab)
glass1.Si=pow(glass1.Si,10)
stats.probplot(glass1.K,dist="norm", plot=pylab)
stats.probplot(pow(glass1.K,1./3.),dist="norm", plot=pylab)
glass1.K=pow(glass1.K,1./3.)
stats.probplot(glass1.Ca,dist="norm", plot=pylab)
stats.probplot(np.log(glass1.Ca),dist="norm", plot=pylab)
glass1.Ca=np.log(glass1.Ca)
stats.probplot(glass1.Ba,dist="norm", plot=pylab)
stats.probplot(pow(glass1.Ba,5),dist="norm", plot=pylab)
glass1.Ba=pow(glass1.Ba,10)
stats.probplot(glass1.Fe,dist="norm", plot=pylab)
#feature selection
output_column = glass1['Type']  # Extract the output column
features = glass1.drop(columns=['Type'])
minmaxscale = MinMaxScaler()
# Fit and transform the data (excluding the output column)
features_scaled = minmaxscale.fit_transform(features)
dataset1 = pd.DataFrame(features_scaled, columns=features.columns)
# Add the output 
dataset1['Type'] = output_column.values
res1 = dataset1.describe()
print(res1)
#all the preprocessing stepps are complted
#we move on to model building
#we are using knn classifier
X=np.array(dataset1.drop(columns=['Type']))
Y=np.array(dataset1['Type'])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
X_train.shape
X_test.shape
Y_train.shape
X_train=np.array(X_train)
X_test=np.array(X_test)
train_acc=[]
test_acc=[]
best_fit=[]
conf_matrices=[]
for k in range(1,214):
    if k%7==0:
        continue
    else:
        knn=KNeighborsClassifier(n_neighbors=k)
        ans=knn.fit(X_train,Y_train)
        pred_train=knn.predict(X_train)
        train_acc.append(skmet.accuracy_score(Y_train, pred_train))
        pred_test=knn.predict(X_test)
        test_acc.append(skmet.accuracy_score(Y_test, pred_test))
        best_fit.append(abs(skmet.accuracy_score(Y_train, pred_train)-skmet.accuracy_score(Y_test, pred_test)))
        conf_matrix = skmet.confusion_matrix(Y_test, pred_test)
        conf_matrices.append(conf_matrix)
#this model got its best_fit at k=8 where train_acc is 75.87 and test acc is 69.65
cor=dataset1.corr()

        
        