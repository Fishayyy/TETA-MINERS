'''
Lab 6
'''


########## Part 1 ###########

'''
    1)  from sklearn.datasets import load_boston
    Extract the description of all the features and print it
    Split your data into train(80% of data) and test(20% of data) via random selection      
'''
# YOUR CODE GOES HERE  
from sklearn.datasets import load_boston
X= load_boston()
print(X.feature_names)


'''
    2)  Try LinearRegression from sklearn.linear_model   
        Try it with and without normalization. Compare the results and pick the best trained model(for comparisson try different metrics from sklearn.metrics like: r2, mse, mae)
        (Hint: for normalizing your data set normalize=True)
    
'''
# YOUR CODE GOES HERE  

'''
    3) Write the equation of the resulted hyper-plane in Q2.
'''



'''
    4)  Repeat Q2 with KNeighborsRegressor. Tune the hyper-parameters(e.g. n_neighbors & metric) using cv techniques. 
'''
# YOUR CODE GOES HERE  


'''
    5) Repeat Q2 with DecisionTreeRegressor from sklearn.tree. Tune the hyper-parameters (e.g. criterion) using cv techniques.
    
'''
# YOUR CODE GOES HERE  

'''
    6) Which model performs better on the test data?
    
'''

########## Part 2 ###########

'''
    1)  Repeat part 1 with Normalized data. (Hint: use standarscalar from sklearn)
'''
# YOUR CODE GOES HERE  