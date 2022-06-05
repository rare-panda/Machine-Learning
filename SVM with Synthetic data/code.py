# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH DATA GENERATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
# Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30) # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
    y[ind] = 1 - y[ind]
    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

#
# DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION,
# MAKE A COPY OF THIS FUNCTION AND THEN EDIT
#
def visualize(models, param, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1
    
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])
    # Create a mesh
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),np.arange(yMin, yMax, 0.01))
    for i, (p, clf) in enumerate(models.items()):
        # if i > 0:
        # break
        r, c = np.divmod(i, 3)
        ax = axes[r, c]
        # Plot contours
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)
        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))

# Generate the data
n_samples = 300 # Total size of data set
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

# ----------------------------------------------------------------------------
# ------------------------------PART A----------------------------------------

# Learn support vector classifiers with a radial-basis function kernel with
# fixed gamma = 1 / (n_features * X.std()) and different values of C
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)

models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()

for C in C_values:
    models[C] = SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr',
                    degree=3, gamma='scale', kernel='rbf',
                    max_iter=-1, probability=False,random_state=None, 
                    shrinking=True,tol=0.001, verbose=False)
    fit = models[C].fit(X_trn, y_trn)

    
    trnErr[C] = 1 - fit.score(X_trn, y_trn)
    valErr[C] = 1 - fit.score(X_val, y_val)
    tstErr[C] = 1 - fit.score(X_tst, y_tst)
visualize(models, 'C', X_trn, y_trn)
C_BstVal = min(valErr, key=valErr.get)
print('Best value of C is ',C_BstVal)
yPred = models[C_BstVal].predict(X_tst)
accuracy = accuracy_score(y_tst, yPred)

print("Accuracy of the model for the best value of C is: ",accuracy)

plt.figure()
plt.grid()
plt.xscale('log')
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.plot(list(tstErr.keys()), list(tstErr.values()), marker='x', linewidth=3, markersize=12)
plt.xlabel('C', fontsize=12)
plt.ylabel('Validation/Training Error', fontsize=12)
plt.xticks(list(valErr.keys()), fontsize=10)
plt.legend(['Validation Error', 'Training Error'], fontsize=12)

# ----------------------------------------------------------------------------
# ------------------------------PART B----------------------------------------

# Learn support vector classifiers with a radial-basis function kernel with
# fixed C = 10.0 and different values of gamma
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)

models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()
for G in gamma_values:
    models[G]=SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', 
                   degree=3, gamma=G, kernel='rbf',
                   max_iter=-1, probability=False, 
                   random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
    fit = models[G].fit(X_trn, y_trn)

    trnErr[G] = 1 - fit.score(X_trn, y_trn)
    valErr[G] = 1 - fit.score(X_val, y_val)
    tstErr[G] = 1 - fit.score(X_tst, y_tst)
visualize(models, 'gamma', X_trn, y_trn)
bestValue = min(valErr, key=valErr.get)
print("Best value of G is ",bestValue)
yPred = models[bestValue].predict(X_tst)
accuracy = accuracy_score(y_tst, yPred)
print("Accuracy of the model for the best value of G is: ",accuracy)

plt.figure()
plt.grid()
plt.xscale('log')
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.plot(list(tstErr.keys()), list(tstErr.values()), marker='x', linewidth=3, markersize=12)
plt.xlabel('G', fontsize=12)
plt.ylabel('Validation/Training error', fontsize=10)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=12)

# ----------------------------------------------------------------------------
# ------------------------------ Q2  ----------------------------------------
# Load the Breast Cancer Diagnosis data set; download the files from eLearning
# CSV files can be read easily using np.loadtxt()
#
# Insert your code here.
#

BCD_trn = np.loadtxt('wdbc_trn.csv', delimiter = ',')
BCD_tst = np.loadtxt('wdbc_tst.csv', delimiter = ',')
BCD_val = np.loadtxt('wdbc_val.csv', delimiter = ',')

X_trn = np.array(BCD_trn[:,1:])
y_trn = np.array(BCD_trn[:,0])
X_tst = np.array(BCD_tst[:,1:])
y_tst = np.array(BCD_tst[:,0])
X_val = np.array(BCD_val[:,1:])
y_val = np.array(BCD_val[:,0])

#
#
# Insert your code here to perform model selection
#
#
import pprint
C_range = np.arange(-2.0, 5.0, 1.0)
C_vals = np.power(10.0, C_range)
gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_vals = np.power(10.0, gamma_range)

models = dict()
trn_err = dict()
val_err = dict()
tstErr = dict()

accuracy = 0
C_BstVal = 0
bestGammaValue = 0


for C in C_vals:
    for G in gamma_vals:
        models[(C,G)]=SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, 
                          gamma=G, kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)
        
        fit = models[(C,G)].fit(X_trn, y_trn)
              
        trn_err[(C,G)] = 1 - fit.score(X_trn, y_trn)
        val_err[(C,G)] = 1 - fit.score(X_val, y_val)


bestVal = min(val_err, key=val_err.get)
print("Best values of C and Gamma are given by",bestVal,"respectively")
yPred = models[bestVal].predict(X_tst)
accuracy = accuracy_score(y_tst, yPred)
print("Accuracy of the model for the best values of C and G is given by: ",accuracy)
pp = pprint.PrettyPrinter(indent=4)
print("Training Error Table")
pp.pprint(trn_err)

print("Validation Error Table") 
pp.pprint(val_err) 

# ----------------------------------------------------------------------------
# ------------------------------ Q3  -----------------------------------------

from sklearn.neighbors import KNeighborsClassifier

k_values = {1,5,11,15,21}

models_k = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')     
    c = neigh.fit(X_trn, y_trn)                                              
    models_k.update({k:c})                 
    s_trn = c.score(X_trn, y_trn)
    trnErr.update({k:1-s_trn})
    s_val = c.score(X_val,y_val)
    valErr.update({k:1-s_val})
    s_tst = c.score(X_tst,y_tst)
    tstErr.update({k:1-s_tst})

#Plotting the training error and validation errors
plt.figure()
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('K - values')
plt.ylabel('Validation/Test error')
plt.xticks(list(valErr.keys()))
plt.legend(['Validation Error', 'Trn Error'])
plt.title("Error Plotting")


#Model Selection based on the best k value
dif_Err = {key: abs(trnErr[key] - valErr.get(key, 0)) for key in trnErr.keys()}
#print(dif_Err)
BestValue_K= min(dif_Err, key=lambda k: dif_Err[k])         
print("Best K value is:",BestValue_K)                        

#The accuracy on test Set

m = models_k[BestValue_K]             

Acc_tst =m.score(X_tst,y_tst)         
print("Accuracy on Test Set is:",Acc_tst*100)