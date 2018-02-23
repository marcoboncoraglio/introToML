import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# read csv data
data = pd.read_csv("ENB2012_data.CSV", sep=';')
data = data.iloc[:, :10]

rowsTest = int(round(data['X1'].count() * 0.8))
trainData_X = data.iloc[:rowsTest, :8]
trainData_Y = data.iloc[:rowsTest, 8:]
# print("Train Data:")
# print(trainData_X)
# print(trainData_Y)
testData_X = data.iloc[rowsTest:, :8]
testData_Y = data.iloc[rowsTest:, 8:]
# print("Test Data:")
# print(testData_X)
# print(testData_Y)

"""
Specifically:
X1 Relative Compactness
X2 Surface Area
X3 Wall Area
X4 Roof Area
X5 Overall Height
X6 Orientation
X7 Glazing Area
X8 Glazing Area Distribution
y1 Heating Load
y2 Cooling Load
"""

degrees = [1, 2, 3, 4, 5, 6, 7, 8]
scorePoly = []
print("\nPolynomial Regression:")
for i in range(len(degrees)):
    linear_regression = LinearRegression()
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    # Train Data
    pipeline.fit(trainData_X, trainData_Y)
    # Create prediction data using testing set
    predData_Y = pipeline.predict(testData_X)
    # append new Variance Score on Score Array for Analytics
    scorePoly.append(r2_score(testData_Y, predData_Y))

    print("Degree: ", i + 1)
    # The coefficients
    # print('Coefficients: \n', linear_regression.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(testData_Y, predData_Y))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(testData_Y, predData_Y))

alpha = [0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]
scoreRidge = []
print("\nRidge:")
for i in range(len(alpha)):
    modelRidge = linear_model.Ridge(alpha=alpha[i])
    # Train Data
    modelRidge.fit(trainData_X, trainData_Y)
    # Create prediction data using testing set
    predData_Y = modelRidge.predict(testData_X)
    # append new Variance Score on Score Array for Analytics
    scoreRidge.append(r2_score(testData_Y, predData_Y))

    print("Degree: ", i + 1)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(testData_Y, predData_Y))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(testData_Y, predData_Y))

# Polynomial Degree Plot
plt.subplot(121)
plt.title("Polynomial Degree")
plt.xlabel("Degree")
plt.ylabel("Variance Score")
plt.scatter(degrees, scorePoly, color='blue', s=4)
plt.plot(degrees, scorePoly, color='blue', linewidth=0.8, label="Linear Regression")

# Ridge Plot
plt.subplot(122)
plt.title("Ridge")
plt.xlabel("Alpha")
plt.ylabel("Variance Score")
plt.scatter(alpha, scoreRidge, color='orange', s=4)
plt.plot(alpha, scoreRidge, color='orange', linewidth=0.8, label="Ridge")

plt.tight_layout()
plt.show()
