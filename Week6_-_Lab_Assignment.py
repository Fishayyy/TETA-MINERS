'''
Lab 6
'''
from random import randint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

seed = input("Enter a Random Seed #: ")

try:
   RANDOM_SEED = int(seed)
except ValueError:
   RANDOM_SEED = randint(1,9999)
   print(f"\'{seed}\' is an invalid choice. Using {RANDOM_SEED} insead.")

########## Part 1 ###########

'''
    1)  from sklearn.datasets import load_boston
    Extract the description of all the features and print it
    Split your data into train(80% of data) and test(20% of data) via random selection      
'''
# YOUR CODE GOES HERE  
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X = load_boston()
y = X['target']
print(X['feature_names'])
print(X['DESCR'])

X_train, X_test, y_train, y_test = train_test_split(X['data'],y, test_size=0.2, random_state=RANDOM_SEED)

'''
    2)  Try LinearRegression from sklearn.linear_model   
        Try it with and without normalization. Compare the results and pick the best trained model(for comparisson try different metrics from sklearn.metrics like: r2, mse, mae)
        (Hint: for normalizing your data set normalize=True)

        Link: https://stackoverflow.com/questions/54067474/comparing-results-from-standardscaler-vs-normalizer-in-linear-regression    
'''
# YOUR CODE GOES HERE 
from sklearn.linear_model import LinearRegression

reg = LinearRegression(normalize=False)
reg.fit(X_train, y_train)

reg_pred = reg.predict(X_test)
print("Linear Regression score: ", reg.score(X_test, y_test)*100)

reg_norm = LinearRegression(normalize=True)
reg_norm.fit(X_train, y_train)

reg_norm_pred = reg_norm.predict(X_test)

'''
    3) Write the equation of the resulted hyper-plane in Q2.
'''

print(f"\nCoefficients: {reg_norm.coef_}")
print(f"Intercept: {reg_norm.intercept_}")
equation = "\ny = "

for i in range(len(reg_norm.coef_)):
    equation += f"({reg_norm.coef_[i]})b_{i+1} + "

equation += f"{reg_norm.intercept_}\n"

print(equation)

'''
    4)  Repeat Q2 with KNeighborsRegressor. Tune the hyper-parameters(e.g. n_neighbors & metric) using cv techniques. 
'''
# YOUR CODE GOES HERE
k = [1, 3, 5, 7, 9, 11, 13, 15]
neighbors = {}
for n in k:
    neigh = KNeighborsRegressor(n_neighbors=n ,p=1)
    neigh.fit(X_train,y_train)
    pred_KNRegressor = neigh.predict(X_test)
    neighbors[n] = pred_KNRegressor
    print(neigh.score(X_test, y_test)*100)


'''
    5) Repeat Q2 with DecisionTreeRegressor from sklearn.tree. Tune the hyper-parameters (e.g. criterion) using cv techniques.
    
'''
# YOUR CODE GOES HERE

decision = DecisionTreeRegressor(criterion="mae")
decision.fit(X_train,y_train)
pred_KNRegressor = decision.predict(X_test)
print("\n")
print(decision.score(X_test, y_test)*100)

'''
    6) Which model performs better on the test data?
    
'''

########## Part 2 ###########

'''
    1)  Repeat part 1 with Normalized data. (Hint: use standarscalar from sklearn)
'''
# YOUR CODE GOES HERE  
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(X.data)
X_norm = scalar.transform(X.data)

#Normalized DATA
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=RANDOM_SEED)


Linear_regression = LinearRegression(normalize=False)
Linear_regression.fit(X_train, y_train)
LRpred = Linear_regression.predict(X_test)
print("Linear Regression with Normalize Data score: ",Linear_regression.score(X_test, y_test)*100)

Linear_regressionNorm = LinearRegression(normalize=True)
Linear_regressionNorm.fit(X_train, y_train)
LRNpred = Linear_regressionNorm.predict(X_test)
print("Linear Regression Norm with Normalize Data score: ",Linear_regressionNorm.score(X_test, y_test)*100)




k = [1, 3, 5, 7, 9, 11, 13, 15]
neighbors = {}
for n in k:
    neigh = KNeighborsRegressor(n_neighbors=n ,p=1)
    neigh.fit(X_train,y_train)
    pred_KNRegressor = neigh.predict(X_test)
    neighbors[n] = pred_KNRegressor
    print(neigh.score(X_test, y_test)*100)

decision = DecisionTreeRegressor(criterion="mae")
decision.fit(X_train,y_train)
pred_KNRegressor = decision.predict(X_test)
print("\n")
print(decision.score(X_test, y_test)*100)

