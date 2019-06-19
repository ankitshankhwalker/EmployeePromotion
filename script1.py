# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:51:49 2019

@author: Ankit
data preparation and exploratory data analysis

"""


import pandas as pd
from fancyimpute import KNN

Traindata = pd.read_csv('TrainData.csv')

#Missing data
Total = Traindata.isnull().sum().sort_values(ascending=False)
Percent = (Traindata.isnull().sum().sort_values(ascending=False)/Traindata.isnull().count().sort_values(ascending=False))*100
missing_data = pd.concat([Total,Percent],axis=1,keys=['Total','Percent'])
missing_data = missing_data[missing_data['Total']>0]

#replacing missing data values
#fancy impute removes column names.
#Check this section as it is not running#
train_cols = list(Traindata)
train_complete = pd.DataFrame(KNN(k=5).complete(Traindata))
train_complete = pd.DataFrame(KNN(k=5).fit_transform(Traindata[['education']]))

#check the distribution of variables post imputation


