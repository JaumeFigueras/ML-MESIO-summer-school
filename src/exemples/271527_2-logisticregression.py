########################################################################################################
import numpy as np
import pandas as pd

Feature = [50, 55, 70, 90, 87, 112, 96, 75, 67, 72, 84, 92, 101, 98]  # People weights in Kg
TARGET = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1]  # TARGET = 1 -> Diabetes

train = pd.DataFrame({'weight': Feature, 'diabetes': TARGET})

# 1) LOGISTIC REGRESSION from package:
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1E20, solver='lbfgs')  # Very high C to obtain an standard Logisic Regression
model.fit(X=train[['weight']], y=train['diabetes'])

print('Intercept:\t',model.intercept_[0],'\nCoef:\t\t',model.coef_[0][0])


from sklearn.metrics import roc_auc_score

results = train.copy()
results['predictions'] = model.predict_proba(train[['weight']])[:, 1]
print('AUC: ', roc_auc_score(results['diabetes'], results['predictions']))
print('GINI: ', 2 * roc_auc_score(results['diabetes'], results['predictions']) - 1)


# 2) LOGISTIC REGRESSION Manually:

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


def net_input(betas, x):
    # Computes the weighted sum of inputs: sum_i(Beta0+Beta1*x_i)
    return np.dot(x, betas)


def probability(betas, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(betas, x))


def cost_function(betas, x, y):
    # Computes the cost function for all the training samples
    total_cost = -np.sum(
        y * np.log(probability(betas, x)) + (1 - y) * np.log(1 - probability(betas, x)))  # Negative we want to minimize
    return total_cost


# Initialize with the feature and TARGET
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

# Transform to arrays in a suitable shape
x = np.c_[np.ones((X.shape[0], 1)), X]  # <---- X=(x1,...,xm) in column to ((1,x1),...,(1,xm)) in column
y = np.array(y)[:, np.newaxis]  # <---- Converts to y=(y1,...,ym) in column as numpy array
betas = np.zeros((x.shape[1], 1))  # <---- initialize betas to betas=(0,0) in column

# MINIMIZE COST FUNCTION
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_tnc.html
from scipy.optimize import fmin_tnc

output = fmin_tnc(func=cost_function, x0=betas, approx_grad=True, args=(x, y.flatten()))
# ----> returns the betas [beta0,beta1]=[intercept,weight] of the minimum of the cost function

beta_vector = output[0]
intercept = beta_vector[0]
weight = beta_vector[1]

print('Intercept:\t',intercept,'\nCoef:\t\t',weight)

# Manually make predictions:
results['predictions_manual'] = 1 / (1 + np.exp(-(intercept + weight * results['weight'])))
# Or using defined functions
results['predictions_manual2'] = probability(beta_vector, x)

print('AUC: ', roc_auc_score(results['diabetes'], results['predictions_manual']))
print('GINI: ', 2 * roc_auc_score(results['diabetes'], results['predictions_manual']) - 1)
