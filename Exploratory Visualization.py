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


fig = plt.figure(figsize = (18,12), dpi=100)
a=0.65
# Step 1
ax11 = fig.add_subplot(341)
df.Survived.value_counts().plot(kind='bar', color="blue", alpha=a)



ax11.set_xticklabels(["Died","Survived"], rotation=0)
plt.title("Overall",fontsize=font_size)

# Step 2
ax21 = fig.add_subplot(345)
df.Survived[df.Sex == 'male'].value_counts(sort=False).plot(kind='bar',label='Male')
##df.Survived[df.Sex == 'female'].value_counts().plot(kind='bar', color='#FA2379',label='Female')
ax21.set_xticklabels(["Died","Survived"], rotation=0)
plt.title("\nWho Survied? with respect to Gender.",fontsize=font_size); plt.legend(loc='best')


ax22 = fig.add_subplot(346)
df.Survived[df.Sex == 'female'].value_counts(sort=False).plot(kind='bar', color='#FA2379',label='Female')
##df.Survived[df.Sex == 'male'].value_counts().plot(kind='bar',label='Male',alpha = 0.5)
##df.Survived[df.Sex == 'female'].value_counts(sort=False).plot(kind='bar', color='#FA2379',label='Female',alpha = 0.5)
ax22.set_xticklabels(["Died","Survived"], rotation=0)
plt.title("\nWho Survied? with respect to Gender.",fontsize=font_size); plt.legend(loc='best')

ax23 = fig.add_subplot(347)
##(df.Survived[df.Sex == 'male'].value_counts()/float(df.Sex[df.Sex == 'male'].size)).plot(kind='bar',label='Male')
(df.Survived[df.Sex == 'female'].value_counts(sort=False)/float(df.Sex[df.Sex == 'female'].size)).plot(kind='bar', color='#FA2379',label='Female')
ax23.set_xticklabels(["Died","Survived"], rotation=0)
plt.title("Who Survied proportionally?",fontsize=font_size); plt.legend(loc='best')

ax24 = fig.add_subplot(348)
(df.Survived[df.Sex == 'male'].value_counts(sort=False)/float(df.Sex[df.Sex == 'male'].size)).plot(kind='bar',label='Male')
##(df.Survived[df.Sex == 'female'].value_counts()/float(df.Sex[df.Sex == 'female'].size)).plot(kind='bar', color='#FA2379',label='Female')
ax24.set_xticklabels(["Died","Survived"], rotation=0)
plt.title("Who Survied proportionally?",fontsize=font_size); plt.legend(loc='best')

#Step 3
ax1=fig.add_subplot(349)
df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts(sort=False).plot(kind='bar', label='female highclass', color='#FA2479', alpha=a)
ax1.set_xticklabels(["Died","Survived"], rotation=0)
##plt.title("Step. 3",fontsize=font_size);

plt.legend(loc='best')


ax2=fig.add_subplot(3,4,10, sharey=ax1)
df.Survived[df.Sex=='female'][df.Pclass==3].value_counts().plot(kind='bar', label='female, low class', color='pink', alpha=a)
ax2.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')

ax3=fig.add_subplot(3,4,11, sharey=ax1)
df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts().plot(kind='bar', label='male, low class',color='lightblue', alpha=a)
ax3.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')

ax4=fig.add_subplot(3,4,12, sharey=ax1)
df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts().plot(kind='bar', label='male highclass', alpha=a, color='steelblue')
ax4.set_xticklabels(["Died","Survived"], rotation=0)
plt.legend(loc='best')

plt.show()
##plt.savefig("Exploratory Visualization.png", dpi=100)
