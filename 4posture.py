#Import Library
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 

xy =np.loadtxt('4posture.csv',delimiter=',',dtype=np.int32)

x_vals =xy[:,0:-1]
y_vals =xy[:,[-1]]

y_vals = y_vals.reshape(y_vals.size,1)
y_vals=y_vals.ravel()

print(x_vals.shape)
print(y_vals.shape,y_vals)

x_train, x_test, y_train, y_test = train_test_split(x_vals,y_vals,test_size=0.3, random_state=0)

model = svm.SVC(kernel='rbf', C=100, gamma=0.001) 

model.fit(x_train, y_train)

model.score(x_train, y_train)

#Predict Output
predicted= model.predict(x_test)

print('Number of test: %d, no.error: %d' %(len(x_test),(y_test!=predicted).sum()))
print('Accuracy: %f' %accuracy_score(y_test,predicted))

