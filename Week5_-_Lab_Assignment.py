'''
Lab 5
'''

######### Part 1 ###########
import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
    1-1) Download the iris-data-3 from Canvas, use pandas.read_csv to load it. This dataset has 5 input features: [sepal_length, sepal_width, petal_length, petal_width, color]
    1-2) Find and drop duplicate samples (use "ID" to find duplicates)
    1-3) Split your data into train(70% of data) and test(30% of data) via random selection
'''
# YOUR CODE GOES HERE
seed = input("Enter a Random Seed #: ")

try:
   RANDOM_SEED = int(seed)
except ValueError:
   RANDOM_SEED = randint(1,9999)
   print(f"\'{seed}\' is an invalid choice. Using {RANDOM_SEED} insead.")
   
with open('iris-data-3.csv') as csvfile:
    dff = pd.read_csv(csvfile, delimiter=',')
    dff.drop_duplicates('ID')
    dff = pd.get_dummies(dff, columns=['color'])
    X = dff[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'color_Red', 'color_blue', 'color_pink', 'color_purple','color_red' ]]
    y = (dff["species"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

'''
    2-1) Train a KNN classifier for the (k = 5, metric='manhattan'). Did you encounter an error? what is the error? how can we fix it?
    Fix the error and train your model. (Hint: pandas.get_dummies())
    2-2) Test your classifier with the test set and report the results.
    2-3) Print the confusion matrix for the results on the test set. 
'''
# YOUR CODE GOES HERE
KNeighbors = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
KNeighbors.fit(X_train, y_train)
pred = KNeighbors.predict(X_test)
print(pred)


'''
    3) Test your trained model with the given test set below and report the performance.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE  


'''
    4)  Use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE  


'''
    5)  Use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2
'''
# YOUR CODE GOES HERE  