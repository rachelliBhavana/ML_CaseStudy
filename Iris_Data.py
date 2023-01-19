# In this application we remove one entry from each label of iris dataset and train with
# the remaining entries.
# And we apply predictions based on Decision tree with that removed entries

# Considering below characteristics of Machine Learning Application :
# Classifier : Decision Tree
# DataSet :    Iris Dataset
# Features :   Sepal Width, Sepal Length, Petal Width, Petal Length
# Labels :     Versicolor, Setosa , Virginica
# Training Dataset : 147 Entries
# Testing Dataset : 3 Entries


import numpy as np
from sklearn import tree 
from sklearn.datasets import load_iris

iris=load_iris()

print("Feature Name of Iris data set")
print(iris.feature_names)

print("Target Name of Iris data set")
print(iris.target_names)

# Indices of removed elements
test_index=[1,51,101]

#Training data with removed 
train_target=np.delete(iris.target,test_index)
train_data=np.delete(iris.data,test_index,axis=0)

# Testing data for testing on training data
test_target=iris.target[test_index]
test_data=iris.data[test_index]

# From decision tree classifier
classifier=tree.DecisionTreeClassifier()

# Apply training data to form tree
classifier.fit(train_data,train_target)

print("Values thta we removed from testing")
print(test_target)

print("Result of testing")
print(classifier.predict(test_data))