# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:54:48 2019

@author: Ankit
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font",size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)
from io import StringIO
import requests

url='https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv'

#convert a byte stream to text
s=requests.get(url).text
data = pd.read_csv(StringIO(s))

data['education'].unique()

#grouping together education levels
data['education']=np.where(data['education']=='basic.4y','basic',data['education'])
data['education']=np.where(data['education']=='basic.9y','basic',data['education'])
data['education']=np.where(data['education']=='basic.6y','basic',data['education'])

#data exploration
data['y'].value_counts()
sns.countplot(x='y',data=data,palette='hls')
plt.show()
plt.savefig('DV_count_plot')

#count the percentage contribution of DV values
count_no_sub = len(data[data['y']==0])
count_sub = len(data[data['y']==1])
percentage_of_no_sub = (count_no_sub/(count_sub+count_no_sub))*100
percentage_of_sub = (count_sub/(count_sub+count_no_sub))*100

#data exploration
variable_means_by_DV = data.groupby(['y']).mean()
#analysis results#
#Average age of customers subscribing is higher
#Average last call duration of subscribers is higher
#Days since last customer contact is lower for subscriber
#Campaign - number of contacts made during a campaign are lower
#for subscribers

#visualizations#
#%matplotlib inline
pd.crosstab(data['job'],data['y']).plot(kind='bar')
plt.title('Purchase Frequencey by Job Title')
plt.xlabel('Job Title')
plt.ylabel('Frequency of Purchase')

#above table no providing clear information if the 
#type of job is affecting customer subscription

#modifying the visual
table = pd.crosstab(data['job'],data['y'])
table.div(table.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

#checking the marital status vs subscribers relationship
table1 = pd.crosstab(data['marital'],data['y'])
table1.div(table1.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

#EDA for numerical variables
data.age.hist()
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")