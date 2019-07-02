# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:08:58 2019

@author: Ankit
"""

from imblearn.over_sampling import SMOTE

#Creating Dummy Variables from a dataset
data.dtypes

#create a list of categorical variables
cat_vars = ['job','marital','education','default','housing','loan','contact',
            'month','day_of_week','poutcome']

for var in cat_vars:
    data_temp=pd.get_dummies(data[var],prefix=var)
    data_temp = data.join(data_temp)
    data = data_temp
    
data_vars_retained=data.columns.values.tolist()

#retain only the required columns
data_final = data[list(set(data_vars_retained) - set(cat_vars))]

#rearrangings the columns in a dataframe
column_names=data_final.columns.values.tolist()
column_names = column_names.sort()
data_final = data_final[column_names]

#samples are disproportianate
data_final['y'].value_counts()

#Applying SMOTE - Synthetic Minority Oversampling Technique 
X = data_final.iloc[:,data_final.columns != 'y']
Y = data_final.iloc[:,data_final.columns == 'y']