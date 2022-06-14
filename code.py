import numpy as np
import pandas as pd
import time
import joblib
# plots

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

# load data
df = pd.read_csv('log2.csv')
df.head()

df.describe()

df.isnull().sum()

# target distribution
print('Absolute Frequencies:')
print(df.Action.value_counts())
print()

print('Percentages:')
print(df.Action.value_counts(normalize=True))

df.Action.value_counts().plot(kind='bar')
plt.title('Target (Action)')
plt.grid()
plt.show()



#df['Action'].replace({'allow':0,'deny':1,'drop':2 , 'reset-both' :3}, inplace = True)


X_data =df.drop(columns=['Action'], axis=1)
Y = df['Action']




# Training and Testing
x_train, x_test, y_train, y_test = train_test_split(X_data,Y,test_size = 0.2,random_state=42)
# KNN MODEL TRAINING
knn = KNeighborsClassifier(n_neighbors = 5)  # n_neighbors means k
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print('KNeighborsClassifier Accuracy: {:.2f}%'.format(knn_accuracy*100))




from sklearn.naive_bayes import GaussianNB
NB = GaussianNB() 
NB.fit(x_train, y_train)
NB_pred = knn.predict(x_test)
NB_accuracy = accuracy_score(y_test, NB_pred)
print('Naive Bayes Accuracy: {:.2f}%'.format(NB_accuracy*100))


# providing actual and predicted values
cm = confusion_matrix(y_test, knn_pred)

# If True, write the data value in each cell
sns.heatmap(cm,annot=True)

# saving confusion matrix in png form
plt.savefig('confusion_Matrix.png')
print(cm)

# Exporting the model
joblib.dump(knn, 'savedmodel.pkl') 


