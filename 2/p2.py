import pandas
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

initialDataset = pandas.read_csv("household_power_consumption.txt", sep=";")
dataset = initialDataset.iloc[:, 2:6]

# fix broken data
dataset = dataset.replace(['?'], ['NaN'])
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(dataset)
dataset = pandas.DataFrame(imp.transform(dataset))

# percentage of input csv to be used
numberOfTrain = int(len(dataset)*80/100)
numberOfTest = int(len(dataset)*20/100)

"""
# make train and test data
X: global_active_power(kW), global_reactive_power(kW), voltage (V)
Y: global intensity (A)
"""

powerX_train = dataset.iloc[:numberOfTrain,:3]
powerX_test = dataset.iloc[numberOfTrain:(numberOfTrain+numberOfTest),:3]
powerY_train = dataset.iloc[:numberOfTrain,3]
powerY_test = dataset.iloc[numberOfTrain:(numberOfTrain+numberOfTest),3]

# linear regression
# Create linear regression object
regr = linear_model.LinearRegression()

# Create lasso model object
#regr = linear_model.Lasso(alpha=0.003)

# Train the model using the training sets
regr.fit(powerX_train, powerY_train)

# Make predictions using the testing set
y_pred = regr.predict(powerX_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(powerY_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(powerY_test, y_pred))

#plt.xlabel("test")
#plt.ylabel("predict")
#plt.plot(powerX_test, y_pred)
#plt.show()

"""
# polinomial function
degrees = [1, 2, 3, 4]
score = []

for i in range(len(degrees)):
    print(i+1)
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)

    regr = linear_model.LinearRegression() # Ridge, Lasso, LinearRegression

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", regr)])

    # train
    pipeline.fit(powerX_train, powerY_train)

    # predict
    y_pred = pipeline.predict(powerX_test)
    r2 = r2_score(powerY_test, y_pred)
    score.append(r2)

    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(powerY_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(powerY_test, y_pred))

plt.xlabel("degrees")
plt.ylabel("score")
plt.plot(degrees, score)
plt.show()
"""
