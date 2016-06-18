from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from pybrain.datasets.supervised import SupervisedDataSet
from sklearn.metrics import mean_squared_error as MSE
from patsy import dmatrices,dmatrix
import pandas as pd
import numpy as np
from math import sqrt
import pylab as pl
from scipy import interp
from sklearn import datasets,cross_validation
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc
import matplotlib.pyplot as plt
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series,DataFrame
from pybrain.tools.validation import Validator
font_size = 9


##*******************************Read and Clean data*******************************
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
formula_test = 'C(Pclass) + C(Sex) + Age + SibSp + Parch  + C(Embarked)'

y, X = dmatrices(formula_ml, data=df, return_type='matrix')



mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
'''

##*******************************Cross Validation*******************************
kf = cross_validation.KFold(len(y), n_folds=10)
accuracy = 0.0 
for i, (train, test) in enumerate(kf):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    lenth = len(X_train)
    ds = ClassificationDataSet(9, 1, nb_classes=2)
    for i in range(lenth):
        ds.addSample(X_train[i], [y_train[i]])
    ds._convertToOneOfMany( )
    net = buildNetwork( ds.indim, 5 , ds.outdim, bias = True )
    trainer = BackpropTrainer( net, dataset=ds, momentum=0.1, verbose=True, weightdecay=0.01)
 #   print "training for {} epochs...".format( epochs )

    dstst = ClassificationDataSet(9, 1, nb_classes=2)
    

    for i in range( 20 ):
        trainer.trainEpochs( 5 )
 #   percentError1 = percentError( trainer.testOnClassData(dataset = dstst),dstst['class'] )/100
#    print 'error= ',percentError1
    
    lenthtst = len(X_test)
    dstst = ClassificationDataSet(9, 1, nb_classes=2)
    for i in range(lenthtst):
        dstst.addSample(X_test[i], [y_test[i]])
    dstst._convertToOneOfMany( )
    
    
    p = net.activateOnDataset( dstst )
    result1 = trainer.testOnClassData(dataset = dstst)
    accuracy += 1 - percentError( trainer.testOnClassData(dataset = dstst),dstst['class'] )/100
    
    ##*******************************ROC Curve*******************************
    probas_ = p
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 0])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
print "Mean accuracy of Neural Network Predictions on the data was: " + str(accuracy/10)

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
'''

##*******************************Train data Accuracy*******************************



lenth = len(X)

epochs = 5

ds = ClassificationDataSet(9, 1, nb_classes=2)
for i in range(lenth):
        ds.addSample(X[i], [y[i]])

ds._convertToOneOfMany( )
        
net = buildNetwork( ds.indim, 5 , ds.outdim, bias = True )
trainer = BackpropTrainer( net, dataset=ds, momentum=0.1, verbose=True, weightdecay=0.01)
print "training for {} epochs...".format( epochs )

for i in range(20):
    trainer.trainEpochs( 5 )
p = net.activateOnDataset( ds )
result = trainer.testOnClassData()
accuracy = 1 - percentError( trainer.testOnClassData(),ds['class'] )/100

print "Accuracy of Neural Network Predictions on the Training data was: " + str(accuracy)


##*******************************Train Data Confusion Matrix*******************************

cm = confusion_matrix(y, result)
print cm
target_names = ['0','1']

print(classification_report(y, result, target_names=target_names))
pl.matshow(cm)
pl.title('Train Data Confusion matrix')
pl.colorbar()
pl.ylabel('Actual label')
pl.xlabel('Predicted label')
pl.show()

