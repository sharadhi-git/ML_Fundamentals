#handling a imbalance data set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('kyphosis.csv')
print(data.sample(10))#You can see that data set is not balanced , only 2 presents for kyphosis column

x=data.drop('Kyphosis',axis=1)# create a data set without the Kyphosis col
y=data['Kyphosis']# create a data set with only the Kyphosis col

print(y.value_counts())#shows no of data points for each class.
y.value_counts().plot(kind='bar')#shows a bar chart for two classes
plt.show()

#To run the undersampling,oversampling and SMOTE methods please install the imbalanced-learn library
#pip install imbalanced-learn

#Undersampling method
from imblearn.under_sampling import RandomUnderSampler
undersample=RandomUnderSampler()
x_under,y_under=undersample.fit_resample(x,y)
y_under.value_counts().plot(kind='bar')

#Oversampling
from imblearn.over_sampling import RandomOverSampler
oversample=RandomOverSampler()
x_over,y_over=oversample.fit_resample(x,y)
y_over.value_counts().plot(kind='bar')

##SMOTE
from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_smote,y_smote=smote.fit_resample(x,y)
y_smote.value_counts().plot(kind='bar')

