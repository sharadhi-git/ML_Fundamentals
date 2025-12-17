# Understading outlier detection process
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#upload the data file
data=pd.read_csv('insurance.csv')
data.head()
print(data.shape)

#visualizer to find whether outliers are available
# we are going to use the charges column in the data set to identify what are the outliers
plt.figure()
plt.hist(data['charges'])
plt.xlabel('Charges')
plt.ylabel('Count')
plt.title('Distribution of charges(Before outlier removal)')
plt.show()

#In plot you can see after 50000 frequency is very low, which seems like outliers. Anyway we need to calculate z score to fairly identify outliers

#Generate the z score
mean=np.mean(data['charges'])
std=np.std(data['charges'])
data['charges_zscore']=(data['charges']-mean)/std

#find outliers 
data[data['charges_zscore']>3]
data[data['charges_zscore']<-3]
outlier_indexes=data.index[data['charges_zscore']>3].to_list()
outlier_indexes
outlier_indexes.extend(data.index[data['charges_zscore']<-3].to_list())
outlier_indexes

#dropping outliers
new_data=data.drop(data.index[outlier_indexes])
print(new_data.shape)

##you can see no of rows has reduced
plt.figure()
plt.hist(new_data['charges'])
plt.xlabel('Charges')
plt.ylabel('Count')
plt.title('Distribution of charges(After ouitlier removal)')
plt.show()