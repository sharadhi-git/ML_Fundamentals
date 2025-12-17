#Ths codes provide the understadning for feature selection using mutual information. We can use such feature selection as the basis to run regression or classification models

#First we do the feature selection for a regression problem
import pandas as pd
import matplotlib.pyplot as plt

#to create a dummy data set
from sklearn.datasets import make_regression
x,y = make_regression(n_samples=50,n_features=6)

x=pd.DataFrame(x)
y=pd.DataFrame(y)

print(x.head())
print(y.head())

#To calculate the mutual information between features we use mutual_info_regression
#To select the features accordingly to take higher amount of mutual infor we use SelectKBest
from sklearn.feature_selection import SelectKBest,mutual_info_regression

fs=SelectKBest(score_func=mutual_info_regression,k=3)#select most important 3 features
fs.fit(x,y)
fs.scores_#to show the mutual information of features in values. higher the value higher the importance of that feature

#preparing this fs as a pandas data series
mi_scores=pd.Series(fs.scores_,index=x.columns)

#represent this data series in decending order in a plot
mi_scores.sort_values(ascending=False).plot(kind='bar')

# now create new data set with less features identified as most important three in fs
x_new=fs.fit_transform(x,y)
x_new=pd.DataFrame(x_new)

print(x_new.head())# result in three most important features as column 2,3 and 4 in original data set

#Now we do the feature selection for a classification problem
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_regression

x,y=make_classification(n_samples=50,n_features=6,n_informative=2)
x=pd.DataFrame(x)
y=pd.DataFrame(y)

print(x.head(5))

fs=SelectKBest(score_func=mutual_info_regression,k=2)# select most important 2 features
fs.fit(x,y)

mi_score=pd.Series(fs.scores_,index=x.columns)
mi_score.sort_values(ascending=False).plot(kind='bar')

x_new=fs.fit_transform(x,y)
x_new=pd.DataFrame(x_new)

print(x_new.head(5))
