'''
Lab 6
'''
import numpy as np
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

print("===================PART 1===================")
X = load_boston()
y = X['target']
print(f"Feature Names: {X['feature_names']}")
print("\nDescription")
print("---------------")
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
from sklearn import metrics

lr = LinearRegression(normalize=False)
lr.fit(X_train, y_train)

print("\nLinear Regression Scoring Metrics")
print("----------------------------------")

lr_pred = lr.predict(X_test)
lr_r2 = metrics.r2_score(y_test, lr_pred)
lr_mse = metrics.mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = metrics.mean_absolute_error(y_test, lr_pred)
print("Linear Regression (Non-Normalized): ")
print(f"R^2 Score = {lr_r2}")
print(f"MSE Score = {lr_mse}")
print(f"RMSE Score = {lr_rmse}")
print(f"MAE Score = {lr_mae}")

lr_norm = LinearRegression(normalize=True)
lr_norm.fit(X_train, y_train)

lr_norm_pred = lr_norm.predict(X_test)
lr_norm_r2 = metrics.r2_score(y_test, lr_norm_pred)
lr_norm_mse = metrics.mean_squared_error(y_test, lr_norm_pred)
lr_norm_rmse = np.sqrt(lr_norm_mse)
lr_norm_mae = metrics.mean_absolute_error(y_test, lr_norm_pred)
print("\nLinear Regression (Normalized): ")
print(f"R^2 Score = {lr_norm_r2}")
print(f"MSE Score = {lr_norm_mse}")
print(f"RMSE Score = {lr_norm_rmse}")
print(f"MAE Score = {lr_norm_mae}")

print("\nResults Part 1 - Q2")
print("---------------------")
if lr_rmse < lr_norm_rmse:
    print("Winner: Linear Regression (Non-Normalized)")
    winner = lr
elif lr_rmse > lr_norm_rmse:
    print("Winner: Linear Regression (Normalized)")
    winner = lr_norm
else:
    print("It's a Tie!")
    winner = lr_norm

'''
    3) Write the equation of the resulted hyper-plane in Q2.
'''
print("\nEquation of the Hyperplane")
print("----------------------------")
print(f"Coefficients: {winner.coef_}")
print(f"Intercept: {winner.intercept_}")
equation = f"\n{winner.intercept_}"

for i in range(len(winner.coef_)):
    equation += f" + ({winner.coef_[i]})x_{i+1}"

equation += " = 0\n"

print(equation)

'''
    4)  Repeat Q2 with KNeighborsRegressor. Tune the hyper-parameters(e.g. n_neighbors & metric) using cv techniques. 
'''
# YOUR CODE GOES HERE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()
kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
hp_candidates = [{'n_neighbors': [1,3,5,7,9,11,13,15], 'metric': ['chebyshev', 'euclidean', 'manhattan']}]
grid = GridSearchCV(estimator=knn, param_grid=hp_candidates, cv=kfold, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
print("Part 1 - Q4 + Q5 Results")
print("-------------------------")
print(f"Best KNN: metric={best_knn.metric}, n_neighbors={best_knn.n_neighbors}")

'''
    5) Repeat Q2 with DecisionTreeRegressor from sklearn.tree. Tune the hyper-parameters (e.g. criterion) using cv techniques.
    
'''
# YOUR CODE GOES HERE
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
hp_candidates = [{'criterion':['mse','friedman_mse','mae']}]
grid = GridSearchCV(estimator=dt, param_grid=hp_candidates, cv=kfold, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
print(f"Best DT: criterion={best_dt.criterion}")

'''
    6) Which model performs better on the test data?
    
'''
results = []
lr_pred = lr.predict(X_test)
lr_mse = metrics.mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
results.append((lr, lr_rmse))

lr_norm_pred = lr_norm.predict(X_test)
lr_norm_mse = metrics.mean_squared_error(y_test, lr_norm_pred)
lr_norm_rmse = np.sqrt(lr_norm_mse)
results.append((lr_norm, lr_norm_rmse))

knn_pred = best_knn.predict(X_test)
knn_mse = metrics.mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
results.append((best_knn,knn_rmse))

dt_pred = best_dt.predict(X_test)
dt_mse = metrics.mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
results.append((best_dt, dt_rmse))

print("\nPart 1 - Q6 Results")
print("---------------------")
print(f"Linear Regression: {lr_rmse}")
print(f"Linear Regression Normalized: {lr_norm_rmse}")
print(f"K-Neighbors Regression: {knn_rmse}")
print(f"Decision Tree Regresssion: {dt_rmse}")

results = sorted(results, key=lambda tup:tup[1])

print(f"\nWinner: {results[0][0]}")

########## Part 2 ###########

'''
    1)  Repeat part 1 with Normalized data. (Hint: use standarscalar from sklearn)
'''
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X['data'], y, test_size=0.2, random_state=RANDOM_SEED)
scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

'''
    2)  Try LinearRegression from sklearn.linear_model   
        Try it with and without normalization. Compare the results and pick the best trained model(for comparisson try different metrics from sklearn.metrics like: r2, mse, mae)
        (Hint: for normalizing your data set normalize=True)

        Link: https://stackoverflow.com/questions/54067474/comparing-results-from-standardscaler-vs-normalizer-in-linear-regression    
'''
# YOUR CODE GOES HERE 
lr = LinearRegression(normalize=False)
lr.fit(X_train, y_train)

print("\n===================PART 2===================")
print("Linear Regression Scoring Metrics")
print("----------------------------------")

lr_pred = lr.predict(X_test)
lr_r2 = metrics.r2_score(y_test, lr_pred)
lr_mse = metrics.mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = metrics.mean_absolute_error(y_test, lr_pred)
print("Linear Regression (Non-Normalized): ")
print(f"R^2 Score = {lr_r2}")
print(f"MSE Score = {lr_mse}")
print(f"RMSE Score = {lr_rmse}")
print(f"MAE Score = {lr_mae}")

lr_norm = LinearRegression(normalize=True)
lr_norm.fit(X_train, y_train)

lr_norm_pred = lr_norm.predict(X_test)
lr_norm_r2 = metrics.r2_score(y_test, lr_norm_pred)
lr_norm_mse = metrics.mean_squared_error(y_test, lr_norm_pred)
lr_norm_rmse = np.sqrt(lr_norm_mse)
lr_norm_mae = metrics.mean_absolute_error(y_test, lr_norm_pred)
print("\nLinear Regression (Normalized): ")
print(f"R^2 Score = {lr_norm_r2}")
print(f"MSE Score = {lr_norm_mse}")
print(f"RMSE Score = {lr_norm_rmse}")
print(f"MAE Score = {lr_norm_mae}")

print("\nResults Part 2 - Q2")
print("---------------------")
if lr_rmse < lr_norm_rmse:
    print("Winner: Linear Regression (Non-Normalized)")
    winner = lr
elif lr_rmse > lr_norm_rmse:
    print("Winner: Linear Regression (Normalized)")
    winner = lr_norm
else:
    print("It's a Tie!")
    winner = lr_norm

'''
    3) Write the equation of the resulted hyper-plane in Q2.
'''
print("\nEquation of the Hyperplane")
print("----------------------------")
print(f"Coefficients: {winner.coef_}")
print(f"Intercept: {winner.intercept_}")
equation = f"\n{winner.intercept_}"

for i in range(len(winner.coef_)):
    equation += f" + ({winner.coef_[i]})x_{i+1}"

equation += " = 0\n"

print(equation)

'''
    4)  Repeat Q2 with KNeighborsRegressor. Tune the hyper-parameters(e.g. n_neighbors & metric) using cv techniques. 
'''
# YOUR CODE GOES HERE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

knn = KNeighborsRegressor()
kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
hp_candidates = [{'n_neighbors': [1,3,5,7,9,11,13,15], 'metric': ['chebyshev', 'euclidean', 'manhattan']}]
grid = GridSearchCV(estimator=knn, param_grid=hp_candidates, cv=kfold, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
print("Part 2 - Q4 + Q5 Results")
print("-------------------------")
print(f"Best KNN: metric={best_knn.metric}, n_neighbors={best_knn.n_neighbors}")

'''
    5) Repeat Q2 with DecisionTreeRegressor from sklearn.tree. Tune the hyper-parameters (e.g. criterion) using cv techniques.
    
'''
# YOUR CODE GOES HERE
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
hp_candidates = [{'criterion':['mse','friedman_mse','mae']}]
grid = GridSearchCV(estimator=dt, param_grid=hp_candidates, cv=kfold, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
best_dt = grid.best_estimator_
print(f"Best DT: criterion={best_dt.criterion}")

'''
    6) Which model performs better on the test data?
    
'''
results = []
lr_pred = lr.predict(X_test)
lr_mse = metrics.mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
results.append((lr, lr_rmse))

lr_norm_pred = lr_norm.predict(X_test)
lr_norm_mse = metrics.mean_squared_error(y_test, lr_norm_pred)
lr_norm_rmse = np.sqrt(lr_norm_mse)
results.append((lr_norm, lr_norm_rmse))

knn_pred = best_knn.predict(X_test)
knn_mse = metrics.mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
results.append((best_knn,knn_rmse))

dt_pred = best_dt.predict(X_test)
dt_mse = metrics.mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
results.append((best_dt, dt_rmse))

print("\nPart 2 - Q6 Results")
print("---------------------")
print(f"Linear Regression: {lr_rmse}")
print(f"Linear Regression Normalized: {lr_norm_rmse}")
print(f"K-Neighbors Regression: {knn_rmse}")
print(f"Decision Tree Regresssion: {dt_rmse}")

results = sorted(results, key=lambda tup:tup[1])

print(f"\nWinner: {results[0][0]}")
