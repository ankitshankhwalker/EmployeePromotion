# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:51:49 2019

@author: Ankit
data preparation and exploratory data analysis

"""

import pandas as pd

Train_data=pd.read_csv("TrainData.csv")

#check the summary of data
summary_stats=Train_data.describe()
#--number of trainings not having much variations--#

#checking missing values in data#
#finding the missing values in a column#
missing_values=Train_data.isnull().sum()

education_missing=pd.isnull(Train_data['education'])
education_missing_values=Train_data[education_missing]
