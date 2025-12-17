
import pandas as pd

#Generate a dataset
data=pd.DataFrame({'Math':[70,60,40,80,30],'Cemistry':[60,80,65,55,60],'Maths':[70,60,40,80,30],'Physics':[50,50,50,50,50],'General_Test':[70,70,60,60,80]})
print("Intial data set")
print(data)

#Feature Selection using **variance**
from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold(threshold=0)#select varinace columns which is not 0. means dropping 0 variance columns if any
selected_features=selector.fit_transform(data)
data=pd.DataFrame(selected_features,columns=selector.get_feature_names_out())
print("Data set after dropping the zero variance features")
print(data)# now data not include any 0 variance columns

#Feature selection using **correlation**
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cor=data.corr()
print(cor)

#plot this in a heatmap to visualize correlation
plt.figure(figsize=(10,10))
sns.heatmap(cor,annot=True)
plt.show()

#Now using this cor data frame we find the correlation 1 features. for that we use below code
corr_features=set()
for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i,j])>0.9:
            colname=cor.columns[i]
            corr_features.add(colname)

print(corr_features)

#so we going to drop maths col
data.drop('Maths',axis=1,inplace=True)
print("Data set after droping correlation=1 features")
print(data)

