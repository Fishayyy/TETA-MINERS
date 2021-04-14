'''
Lab 8
'''
print ("Lab 8")

##########Part 0 ###########
'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of a digit)
    breifly explain: what are the features in this classification problem and how many features do we have?
    Find the distribution of the lables. 
    Use plot command to visualize the first five samples in the dataset. What are their lables?
    Split your data into train(80% of data) and test(20% of data) via random selection    

    Answer:
    The features in this classification problem are the pixels and their intensity values. Since the images
    are 8x8 there are 64 features representing each pixel and each feature has a bit-depth of 4 meaning its
    value can range between 0-15 giving each pixel 16 unique shades of grey.
'''
# YOUR CODE GOES HERE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import statistics

digits = load_digits(as_frame = True)

print(f"Label Distribution:\n{digits['target'].value_counts()}\n")
print(f"Mean: {statistics.mean(digits['target'].value_counts())}")
print(f"Standard Deviation: {statistics.pstdev(digits['target'].value_counts())}")
print(f"Variance: {statistics.pvariance(digits['target'].value_counts())}")

for i in range(0,5):
    plt.matshow(digits['images'][i])
    plt.title(f"Digits Image Label - {digits['target'][i]}")
    plt.gray()
    plt.show()
    
X_train, X_test, y_train, y_test = train_test_split(digits['data'], digits['target'], test_size=0.2, random_state=123)

##########Part 1 ###########

'''
    1)  Try LogisticRegression from sklearn.linear_model
        Try to tune the hyperparameters (only change these params: penalty, C, Solver) via hold-out CV (30% for validation).
        candidate values: 
        penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
        solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
        C = [1 , 10 , 0.1]
        
        What is the class_weight param? Do you need to modify that? Why?

        Answer:
        class_weight can help with the training of datasets that are unbalanced by adjusting the weight of a class.

        If provided a dictionary it changes the weights associated with classes in the form {class_label: weight}.
        If the default value of 'None' is given, all classes are supposed to have weight one.
        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class 
        frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        
        However, looking at the statistics I printed out at the begining, you can see that the number of 
        all the samples for each target in this data set are about 180 with small variances. Thus, our
        data set is relatively balanced and we can leave the default of class_weight=None.
'''

# YOUR CODE GOES HERE
from sklearn.linear_model import LogisticRegression

print("\nBegining training on non-normalized data...\n")

penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
C = [1 , 10 , 0.1]
l2_only = ['newton-cg', 'lbfgs']
m_iter = 5000
max_acc = 0

print("Training LogisticRegression...\n")

from sklearn.metrics import accuracy_score
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
for p in penalties:
  for s in solvers:
    if s in l2_only and p != 'l2':
      break
    for c in C:
      if max_acc == 1.0:
        break
      lr = LogisticRegression(penalty=p, C=c, solver=s, max_iter=m_iter)        
      lr.fit(X1, y1)                                      
      pred = lr.predict(X2)
      acc = accuracy_score(y2, pred)
      if acc > max_acc:
        max_acc = acc
        best_LR = lr
        print(f"LR: penalty={p}, C={c}, solver={s}")
        print(f"UPDATE - Current Best Accuracy: {acc}\n")

'''
    2)  Try LinearSVC from sklearn.svm
    Try to tune the hyperparameters (only change these params: penalty, C, loss) via hold-out CV (30% for validation).
    penalties = [ 'l1', 'l2' ]
    C = [1 , 10 , 0.1]
    loss = ['hinge', 'squared_hinge']
'''
# YOUR CODE GOES HERE

from sklearn.svm import LinearSVC
penalties = ['l1', 'l2']
C = [1, 10, 0.1]
loss = ['hinge', 'squared_hinge']
m_iter = 5000
max_acc = 0

print("Training LinearSVC...\n")

for p in penalties:
  for l in loss:
    if l == "hinge" and p == "l1":
      break
    for c in C:
      if max_acc == 1.0:
        break
      lsvc = LinearSVC(penalty=p, loss=l, C=c, class_weight= 'balanced', max_iter=m_iter)  
      lsvc.fit(X1, y1)                                      
      pred = lsvc.predict(X2)
      acc = accuracy_score(y2, pred)
      if acc > max_acc:
        max_acc = acc
        best_LSVC = lsvc
        print(f"Linear SVC: penalty={p}, C={c}, loss={l}")
        print(f"UPDATE - Current Best Accuracy: {acc}\n")

'''
    3)  Try SVC from sklearn.svm (this classifier can also be used with linear kernel == LinearSVC)
    Try to tune the hyperparameters (only change these params: decision_function_shape, C, kernel, degree) via hold-out CV (30% for validation).
    decision_function_shape = [ 'ovo', 'ovr']
    C = [1 , 10, 0.1]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]
    degree = [0, 1, 2, 3, 4, 5, 6]
'''

# YOUR CODE GOES HERE
import numpy as np
from sklearn import svm
decision_function_shape = [ 'ovo', 'ovr']
C = [1 , 10, 0.1]
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]
degree = [0, 1, 2, 3, 4, 5, 6]
precom_train = np.dot(X1, X1.T)
precom_test = np.dot(X2, X1.T)
m_iter = 1000
max_acc = 0

print("Training SVC...\n")

for dfs in decision_function_shape:
  for k in kernel:
    for d in degree:
        if k != 'poly' and d != 3:
                break  
        for c in C:
            if max_acc == 1.0:
                break
            svc = svm.SVC(C = c, kernel=k, decision_function_shape=dfs, degree=d, max_iter=m_iter)
            if k == 'precomputed':
                svc.fit(precom_train, y1)
                pred = svc.predict(precom_test)
            else:
                svc.fit(X1, y1)
                pred = svc.predict(X2)  
            acc = accuracy_score(y2, pred)
            if acc > max_acc:
                max_acc = acc
                best_SVC = svc
                print(f"SVC: decision_function_shape={dfs}, C={c}, kernel={k}, degree={d}")
                print(f"UPDATE - Current Best Accuracy: {acc}\n")

##########Part 2 ###########
'''
    1)  Test your trained models in part1: Q1, Q2, and Q3 with the test set and pick the best model. 
    Try to analyze the confusion matrix and explain which classes are mostly confused with each other.

    Answer:
    It seems that the number 8 is the most common mistake by the classifier as it keeps predicting that it is a 1
    3 and 8 are also confused to a lesser degree in some of the predictions.
    Finally, 5 also noticeably gets confused with 9 from time to time.
'''

# YOUR CODE GOES HERE
from sklearn.metrics import confusion_matrix

print("=============================TEST RESULTS (Non-Normalized)=============================")
pred_LR = best_LR.predict(X_test)
pred_LSVC = best_LSVC.predict(X_test)
pred_SVC = best_SVC.predict(X_test)

acc_LR = accuracy_score(y_test, pred_LR)
best_model = best_LR
best_acc = acc_LR

acc_LSVC = accuracy_score(y_test, pred_LSVC)
if acc_LSVC > best_acc:
    best_model = best_LSVC
    best_acc = acc_LSVC

acc_SVC = accuracy_score(y_test, pred_SVC)
if acc_SVC > best_acc:
    best_model = best_SVC
    best_acc = acc_SVC

print(f"Best Logistic Regression Parameters: penalty={best_LR.penalty}, C={best_LR.C}, solver={best_LR.solver}")
print(f"Logistic Regression Accuracy:{acc_LR}")
print(f"Confusion Matrix for Logistic Regression: \n{confusion_matrix(y_test, pred_LR)}")

print(f"\nBest Linear SVC Parameters: penalty={best_LSVC.penalty}, C={best_LSVC.C}, loss={best_LSVC.loss}")
print(f"Linear SVC Accuracy:{acc_LSVC}")
print(f"Confusion Matrix for Linear SVC: \n{confusion_matrix(y_test, pred_LSVC)}")

print(f"\nBest SVC Parameters: decision_function_shape={best_SVC.decision_function_shape}, C={best_SVC.C}, kernel={best_SVC.kernel}, degree={best_SVC.degree}")
print(f"SVC Accuracy:{acc_SVC}")
print(f"Confusion Matrix for SVC: \n{confusion_matrix(y_test, pred_SVC)}")

print(f"\nWINNER: {best_model}")
print(f"Accuracy: {best_acc}")

##########Part 3 ###########

'''
    1)  Repeat part 1 and 2 with Normalized data

'''

# YOUR CODE GOES HERE
from sklearn.preprocessing import StandardScaler

print("\nBegining training on normalized data...\n")

scalar = StandardScaler()
norm_data = scalar.fit(digits['data']).transform(digits['data'])
X_train, X_test, y_train, y_test = train_test_split(norm_data, digits['target'], test_size=0.2, random_state=123)

penalties = [ 'l1', 'l2', 'elasticnet', 'none' ]
solvers = [ 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga' ]
C = [1 , 10 , 0.1]
l2_only = ['newton-cg', 'lbfgs']
m_iter = 5000
max_acc = 0

print("Training LogisticRegression (Normalized)...\n")

X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
for p in penalties:
  for s in solvers:
    if s in l2_only and p != 'l2':
      break
    for c in C:
      if max_acc == 1.0:
        break
      lr = LogisticRegression(penalty=p, C=c, solver=s, max_iter=m_iter)        
      lr.fit(X1, y1)                                      
      pred = lr.predict(X2)
      acc = accuracy_score(y2, pred)
      if acc > max_acc:
        max_acc = acc
        best_LR = lr
        print(f"LR: penalty={p}, C={c}, solver={s}")
        print(f"UPDATE - Current Best Accuracy: {acc}\n")

penalties = ['l1', 'l2']
C = [1, 10, 0.1]
loss = ['hinge', 'squared_hinge']
m_iter = 100000
max_acc = 0

print("Training LinearSVC (Normalized)...\n")

for p in penalties:
  for l in loss:
    if l == "hinge" and p == "l1":
      break
    for c in C:
      if max_acc == 1.0:
        break
      lsvc = LinearSVC(penalty=p, loss=l, C=c, class_weight= 'balanced', max_iter=m_iter)  
      lsvc.fit(X1, y1)                                      
      pred = lsvc.predict(X2)
      acc = accuracy_score(y2, pred)
      if acc > max_acc:
        max_acc = acc
        best_LSVC = lsvc
        print(f"Linear SVC: penalty={p}, C={c}, loss={l}")
        print(f"UPDATE - Current Best Accuracy: {acc}\n")

decision_function_shape = [ 'ovo', 'ovr']
C = [1 , 10, 0.1]
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]
degree = [0, 1, 2, 3, 4, 5, 6]
precom_train = np.dot(X1, X1.T)
precom_test = np.dot(X2, X1.T)
m_iter = 1000
max_acc = 0

print("Training SVC (Normalized)...\n")

for dfs in decision_function_shape:
  for k in kernel:
    for d in degree:
        if k != 'poly' and d != 3:
                break  
        for c in C:
            if max_acc == 1.0:
                break
            svc = svm.SVC(C = c, kernel=k, decision_function_shape=dfs, degree=d, max_iter=m_iter)
            if k == 'precomputed':
                svc.fit(precom_train, y1)
                pred = svc.predict(precom_test)
            else:
                svc.fit(X1, y1)
                pred = svc.predict(X2)  
            acc = accuracy_score(y2, pred)
            if acc > max_acc:
                max_acc = acc
                best_SVC = svc
                print(f"SVC: decision_function_shape={dfs}, C={c}, kernel={k}, degree={d}")
                print(f"UPDATE - Current Best Accuracy: {acc}\n")

print("=============================TEST RESULTS (Normalized)=============================")
pred_LR = best_LR.predict(X_test)
pred_LSVC = best_LSVC.predict(X_test)
pred_SVC = best_SVC.predict(X_test)

acc_LR = accuracy_score(y_test, pred_LR)
best_model = best_LR
best_acc = acc_LR

acc_LSVC = accuracy_score(y_test, pred_LSVC)
if acc_LSVC > best_acc:
    best_model = best_LSVC
    best_acc = acc_LSVC

acc_SVC = accuracy_score(y_test, pred_SVC)
if acc_SVC > best_acc:
    best_model = best_SVC
    best_acc = acc_SVC

print(f"Best Logistic Regression Parameters: penalty={best_LR.penalty}, C={best_LR.C}, solver={best_LR.solver}")
print(f"Logistic Regression Accuracy:{acc_LR}")
print(f"Confusion Matrix for Logistic Regression: \n{confusion_matrix(y_test, pred_LR)}")

print(f"\nBest Linear SVC Parameters: penalty={best_LSVC.penalty}, C={best_LSVC.C}, loss={best_LSVC.loss}")
print(f"Linear SVC Accuracy:{acc_LSVC}")
print(f"Confusion Matrix for Linear SVC: \n{confusion_matrix(y_test, pred_LSVC)}")

print(f"\nBest SVC Parameters: decision_function_shape={best_SVC.decision_function_shape}, C={best_SVC.C}, kernel={best_SVC.kernel}, degree={best_SVC.degree}")
print(f"SVC Accuracy:{acc_SVC}")
print(f"Confusion Matrix for SVC: \n{confusion_matrix(y_test, pred_SVC)}")

print(f"\nWINNER: {best_model}")
print(f"Accuracy: {best_acc}")
