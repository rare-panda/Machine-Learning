# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 22:40:32 2021

@author: kxc200005
"""

import numpy as np # For all our math needs
import matplotlib.pyplot as plt # For all our plotting needs

n = 750 # Number of data points
X = np.random.uniform(-7.5, 7.5, n) # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n) # Random Gaussian noise


def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y

y = f_true(X) + e

plt.figure()
# Plot the data
plt.scatter(X, y, 12, marker='o')

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')
#------------------------------------------------------------------------------------------------------------------

# Split data points in train and test. Then further split train data points into train, validate:
# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3 # Fraction of examples to sample for the test set
val_frac = 0.1 # Fraction of examples to sample for the validation set
# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
#------------------------------------------------------------------------------------------------------------------

# 1. Part a: Vandermond matrix:
# X float(n, ): univariate data
# d int: degree of polynomial
def polynomial_transform(X, d):
    Phi = []
    for v in X:
        z = []
        for f in range(0, d):
            z.append(np.power(v,f))
        Phi.append(z)   # This will give a 2D list of arrays
    Phi = np.asarray(Phi)   # This will convert 2D list of arrays into array
    return Phi


#------------------------------------------------------------------------------------------------------------------

# 1. Part b: Python function below that takes a Vandermonde matrix and the labels as input and learns
# weights via ordinary least squares regression:
# Phi float(n, d): transformed data
# y float(n, ): labels
def train_model(Phi, y):
    w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    return w

#------------------------------------------------------------------------------------------------------------------

# 1. Part c: the Python function below that takes a Vandermonde matrix , corresponding labels y, and a linear
#regression model as input and evaluates the model using mean squared error:
# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): linear regression model
def evaluate_model(Phi, y, w):
    y_pred = Phi @ w
    err = (y_pred - y)**2
    sum = 0
    for value in err:
        sum = sum + value
    mean_sq_err = sum / n
    return mean_sq_err

#------------------------------------------------------------------------------------------------------------------

# 1. Part d: Train model
w = {}              # Dictionary to store all the trained models
validationErr = {}  # Validation error of the models
testErr = {}        # Test error of all the models

for d in range(3, 25, 3): # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d) # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn) # Learn model on training data

    Phi_val = polynomial_transform(X_val, d) # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d]) # Evaluate model on validation data

    Phi_tst = polynomial_transform(X_tst, d) # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, -10, 60])


plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

# Best fit : d = 24

#------------------------------------------------------------------------------------------------------------------
#----------------------------------PART 2--------------------------------------------
#------------------------------------------------------------------------------------------------------------------

# Regression with Radial Basis Functions
# 2. Part a: function below that takes univariate data as input and computes a radial-basis kernel

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    Phi = []
    for v in X:
        z = []
        for f in B:
            z.append(np.exp(-1 * gamma* ((v - f)**2)))
        Phi.append(z)
    Phi = np.asarray(Phi)
    return Phi


#---------------------------------------------------------------------------------------------------------------------
            
# 2. Part b: Complete the Python function below that takes a radial-basis kernel matrix , the labels , and a regularization
# parameter as input and learns weights via ridge regression.

# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lam):
     w = np.linalg.inv((Phi.T @ Phi) + (lam * np.identity(Phi[0].size))) @ Phi.T @ y
     return w
    
#---------------------------------------------------------------------------------------------------------------------

# 2. Part c: As before, we can explore the tradeoff between fit and complexity by varying lambda
# Plot two curves: lambda vs. validation error and lambda vs. test error, as above
    
w = {}              # Dictionary to store all the trained models
validationErr = {}  # Validation error of the models
testErr = {}        # Test error of all the models

Phi_trn = radial_basis_transform(X_trn, X_trn) # Transform training data into d dimensions

lamrange = [10**x for x in range(-3,4)]
for lam in lamrange: # Iterate over lambda
   
    w = train_ridge_model(Phi_trn, y_trn, lam) # Learn model on training data
    
    Phi_val = radial_basis_transform(X_val, X_trn) # Transform validation data into d dimensions
    validationErr[lam] = evaluate_model(Phi_val, y_val, w) # Evaluate model on validation data
    
    Phi_tst = radial_basis_transform(X_tst, X_trn) # Transform test data into d dimensions
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w) # Evaluate model on test data
    
# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.xscale('log')
    
# Best Fit: lambda = 10^-3 

#---------------------------------------------------------------------------------------------------------------------

# 2. Part d: Plot the learned models as well as the true model similar to the polynomial basis case above

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
lamrange = [10**x for x in range(-3,4)]
for lam in lamrange:
    X_d = radial_basis_transform(x_true, X_trn)
    w = train_ridge_model(X_d, y_true,lam)
    y_d = X_d @ w
    plt.plot(x_true, y_d, marker='None', linewidth=2)
plt.legend(['true'] + lamrange)

# Plot becomes more linear as lambda increases