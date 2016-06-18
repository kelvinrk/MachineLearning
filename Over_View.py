import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.nonparametric import smoothers_lowess

from statsmodels.nonparametric.kde import KDEUnivariate
from pandas import Series,DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

import sys
sys.path.append('C:\Users\Kelvin.R.K\Desktop\RMS Titanic\AGC_KaggleAux-master')
import kaggleaux as ka

font_size = 9

df = pd.read_csv("train.csv")

df = df.drop(['Ticket','Cabin'], axis=1) 

df.Age = df.Age.fillna(df.Age.median())

df.Age = df.Age = (df.Age + 0.99).astype('int') ;

df = df.dropna()

fig = plt.figure(figsize = (18,6), dpi=160) # specifies the parameters of our graphs
a = 0.2                                   # sets the alpha level of the colors in the graph (for more attractive results)
a_bar = 0.55                              # another alpha setting

plt.subplot2grid((2,3),(0,0))             # lets us plot many diffrent shaped graphs together
df.Survived.value_counts().plot(kind='bar', alpha=a_bar)# plots a bar graph of those who surived vs those who did not. 
plt.title("Distribution of Survival, (1 = Survived)",fontsize=font_size) # puts a title on our graph

plt.subplot2grid((2,3),(0,1))
plt.scatter(df.Survived, df.Age, alpha=a)
plt.ylabel("Age",fontsize=font_size)                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.axis([-1, 2, 0, 100])
plt.title("Survial by Age,  (1 = Survived)",fontsize=font_size)


plt.subplot2grid((2,3),(0,2))
df.Pclass.value_counts().plot(kind="barh", alpha=a_bar)
plt.title("Class Distribution",fontsize=font_size)


plt.subplot2grid((2,3),(1,0), colspan=2)
df.Age[df.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age
df.Age[df.Pclass == 2].plot(kind='kde')
df.Age[df.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")                         # plots an axis lable
plt.title("Age Distribution within classes",fontsize=font_size);plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
df.Embarked.value_counts().plot(kind='bar', alpha=a_bar)
plt.title("Passengers per boarding location",fontsize=font_size);

plt.show()
