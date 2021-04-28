
'''
Lab 10
'''
########## Part 1 ###########

'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of a digit)
    Split your data into train(70% of data) and test(30% of data) via random selection.
     
    2)  Try MLPClassifier from sklearn.neural_network
        (a NN with two hidden layers, each with 100 nodes)
        Use 20% of your training data as the validation set to tune other hyper-parameters (e.g. activation, solver). Try different values and pick the best one.
        
    3)  Print classification report for the test set.
'''
# YOUR CODE GOES HERE
import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn as sk
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

RANDOM = 123
# Uncomment if you want to repeat your training results
#tf.random.set_seed(RANDOM) 

digits = load_digits()

X = digits['data']
y = digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM)

#validation set
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM)

activation = ["identity", "logistic", "tanh", "relu"]
solver = ["lbfgs", "sgd", "adam"]
best_MLP = MLPClassifier().fit(X1,y1)
max_acc = 0

for a in activation:
    for s in solver:
        MLP = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=2000, activation=a, solver=s, random_state=RANDOM).fit(X1, y1)
        pred = MLP.predict(X2)
        acc = accuracy_score(y2, pred)
        if acc > max_acc:
            max_acc = acc
            best_MLP = MLP
            print(f"MLP: activation={a}, solver={s}")
            print(f"UPDATE - Current Best Accuracy: {acc}\n")

pred = best_MLP.predict(X_test)
print(classification_report(y_test, pred))

########## Part 2 ###########

'''
    1) Try to have the same NN (the same architecture) in Keras. Try different activation functions for hidden layers to get a reasonable network.
    
    Hint: 
    - use validation set (e.g. 20% of your training data) to pick the best value for the hyper-parameters.
    - you need to convert your labels to vectors of 0s and 1  (Try OneHotEncoder from sklearn.preprocessing)
    - activation fcn for output layer = sigmoid
   
'''

# YOUR CODE GOES HERE
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

enc = OneHotEncoder()
sparse = enc.fit_transform(y.reshape(-1,1))

y = sparse.todense()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM)

#validation set
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM)

model = keras.Sequential()
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X1, y1, batch_size=8, epochs=8)

print("\nSigmoid Output Layer:")
pred = model.predict(X2) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y2, axis=1)
print(classification_report(labels, pred))

'''
    2) Use 'softmax' activation function in output layer, print the predictions/ what is the difference?
'''
# YOUR CODE GOES HERE
print("\nTraining with output layer set to \'softmax\' activation...\n")
model2 = keras.Sequential()
model2.add(layers.Dense(100, activation='relu'))
model2.add(layers.Dense(100, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model2.fit(X1, y1, batch_size=8, epochs=8)

print("\nSoftmax Output Layer:")
pred = model2.predict(X2) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y2, axis=1)
print(classification_report(labels, pred))

# With or without tensorflow's random seed set to 123 for repeatability. It seems that setting softmax
# for the output layer increases the level of accuracy in our training given that the output layer's
# activation function was the only change. Playing with the epochs and batch size of one model seems
# to hurt the other model since the second model seems to learn better with less epochs.

'''
    3) Use a 'dropout' layer (ratio = 0.2) after the second hidden layer, print the predictions.
'''
# YOUR CODE GOES HERE
print("\nAdding dropout layers...\n")
model3 = keras.Sequential()
model3.add(layers.Dense(100, activation='relu'))
model3.add(layers.Dense(100, activation='relu'))
model3.add(layers.Dropout(0.2))
model3.add(layers.Dense(10, activation='softmax'))
model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model3.fit(X1, y1, batch_size=8, epochs=8)

print("\nSoftmax Output Layer with hidden Dropout layer:")
pred = model3.predict(X2) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y2, axis=1)
print(classification_report(labels, pred))

'''
    4) Save your model as a .h5 (or .hdf5) file
    
'''
# YOUR CODE GOES HERE
print("Saving models...\n")
model.save('model_1.h5')
model2.save('model_2.h5')
model3.save('model_3.h5')
'''
    5) Load your saved your model and test it using the test set
    
'''
# YOUR CODE GOES HERE
print('Loading models...\n')
m1 = keras.models.load_model('model_1.h5')
m2 = keras.models.load_model('model_2.h5')
m3 = keras.models.load_model('model_3.h5')

print("Testing models on test data...\n")

print("Sigmoid Model:")
pred = m1.predict(X_test) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y_test, axis=1)
print(classification_report(labels, pred))

print("Softmax Model:")
pred = m2.predict(X_test) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y_test, axis=1)
print(classification_report(labels, pred))

print("Softmax Dropout Model:")
pred = m3.predict(X_test) 
pred = np.argmax(pred, axis=1)
labels = np.argmax(y_test, axis=1)
print(classification_report(labels, pred))