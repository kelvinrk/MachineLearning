import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt

from statsmodels.nonparametric import smoothers_lowess

from statsmodels.nonparametric.kde import KDEUnivariate
from pandas import Series,DataFrame
from patsy import dmatrices,dmatrix
from sklearn import datasets, svm, cross_validation, preprocessing    
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pylab as pl

import sklearn.ensemble as ske
from scipy import interp

font_size = 9

##Read and Clean data
df = pd.read_csv("train.csv")

df = df.drop(['Ticket','Cabin'], axis=1) 

df.Age = df.Age.fillna(df.Age.median())

df.Age = df.Age = (df.Age + 0.99).astype('int') ;

df = df.dropna()


test_data = pd.read_csv("test.csv")


test_data = test_data.drop(['Ticket','Cabin'], axis=1) 

test_data.Age = test_data.Age.fillna(test_data.Age.median())

test_data.Age = test_data.Age = (test_data.Age + 0.99).astype('int') ;

a = pd.isnull(test_data.Fare);

fare = test_data.Fare.fillna(test_data[(test_data['Embarked']==str(test_data[a].Embarked)[7]) & (test_data['Pclass'] ==int(test_data[a].Pclass))].Fare.median());

test_data.Fare = fare;

test_data = test_data.dropna()



# Create an acceptable formula for our machine learning algorithms
formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'

formula_test = 'C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'


# create a regression friendly data frame
y, x = dmatrices(formula_ml, data=df, return_type='matrix')
X = np.asarray(x)
y = np.asarray(y)
y = y.flatten()


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []


##*******************************Cross Validation*******************************
kf = cross_validation.KFold(len(y), n_folds=10)
accuracy = 0.0
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=2,
    gamma=1.0, kernel='rbf', max_iter=-1, probability=True,
    random_state=None, shrinking=True, tol=0.001, verbose=False)


for i, (train, test) in enumerate(kf):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
    accuracy += clf.score(X_test, y_test)
    
##*******************************ROC Curve*******************************
    probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
print "Mean accuracy of SVM Predictions of Cross Validation was: " + str(accuracy/10)


pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(kf)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
print("Area under the ROC curve : %f" % mean_auc)
pl.plot(mean_fpr, mean_tpr, 'k--',
        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

pl.xlim([-0.05, 1.05])
pl.ylim([-0.05, 1.05])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic Curve')
pl.legend(loc="lower right")
pl.show()




##*******************************Predict for Real Test data*******************************    
xt = dmatrix(formula_test, data=test_data, return_type='matrix')
XT = np.asarray(xt)

res_svm = clf.predict(XT)

res_svm = DataFrame(res_svm,columns=['Survived'])
res_svm.to_csv("SVM Result.csv")

##*******************************Train data Accuracy*******************************
train_prediction = clf.fit(X, y).predict(X)
result_train = clf.score(X, y)
print "Accuracy of SVM Predictions on the Training data was: " + str(result_train)

##*******************************Train Data Confusion Matrix*******************************
cm = confusion_matrix(y, train_prediction)
print cm
target_names = ['0','1']

print(classification_report(y, train_prediction, target_names=target_names))
pl.matshow(cm)
pl.title('Train Data Confusion matrix')
pl.colorbar()
pl.ylabel('Actual label')
pl.xlabel('Predicted label')
pl.show()


##*******************************Test data Accuracy*******************************
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

##clf.fit(X_train, y_train)
test_prediction = clf.fit(X_train, y_train).predict(X_test)

right = sum(test_prediction==y_test)

result_test = 1.0*right/len(X_test)

print "Accuracy of SVM Predictions on the Test data was: " + str(result_test)


##*******************************Test Data Confusion Matrix*******************************
cm = confusion_matrix(y_test, test_prediction)
print cm
target_names = ['0','1']

print(classification_report(y_test, test_prediction, target_names=target_names))
pl.matshow(cm)
pl.title('Test Data Confusion matrix')
pl.colorbar()
pl.ylabel('Actual label')
pl.xlabel('Predicted label')
pl.show()








