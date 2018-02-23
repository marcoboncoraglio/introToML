import pandas
from io import open
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer, PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

#initialDataset = pandas.read_csv("household_power_consumption.txt", sep=";")
initialDataset = pandas.read_csv("ENB2012_data.CSV", sep=";")
dataset = initialDataset.iloc[:, :10]

# fix broken data
#dataset = dataset.replace(['?'], ['NaN'])
#imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit(dataset)
#dataset = pandas.DataFrame(imp.transform(dataset))

# percentage of input csv to be used
numberOfTrain = int(len(dataset)*80/100)
numberOfTest = int(len(dataset)*20/100)

"""
# make train and test data
X: global_active_power(kW), global_reactive_power(kW), voltage (V)
Y: global intensity (A)
"""

powerX_train = dataset.iloc[:numberOfTrain,:8]
powerX_test = dataset.iloc[numberOfTrain:(numberOfTrain+numberOfTest),:8]
powerY_train = dataset.iloc[:numberOfTrain,8:]
powerY_test = dataset.iloc[numberOfTrain:(numberOfTrain+numberOfTest),8:]

def linearRegression(type):
    # Create linear regression object
    if(type == "lr"):
        regr = linear_model.LinearRegression()
    elif(type == "ridge"):
        regr = linear_model.Ridge(alpha=3)
    elif(type == "lasso"):
        regr = linear_model.Lasso(alpha=1)
    else:
        print("wrong parameter")
        return

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

def polinomialFunction():
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
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
        print('Variance score: %.2f' % r2)

    plt.xlabel("degrees")
    plt.ylabel("score")
    plt.title("Polynomial Regression")
    ax = plt.plot(degrees, score)
    plt.show()

# finds best alphas for ridge and lasso in function of variance
# rigde, ridgeCV = all alphas give same variance, lasso = 0.003, lassoCV = 0.00001
def alphaViz():
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 1, 3, 10]
    optimizationFunctions = ['ridge','lasso', 'lassolars']
    score = []

    f, axarr = plt.subplots(len(optimizationFunctions))
    funcIndex = 0
    for func in optimizationFunctions:
        for i in alphas:
            print(func,i)

            # no switch case in python :(
            if func == 'ridge':
                regr = linear_model.Ridge(alpha=i)
            elif func == 'lasso':
                regr = linear_model.Lasso(alpha=i)
            elif func == 'lassolars':
                regr = linear_model.LassoLars(alpha=i)

            # Train the model using the training sets
            regr.fit(powerX_train, powerY_train)

            # Make predictions using the testing set
            y_pred = regr.predict(powerX_test)
            r2 = r2_score(powerY_test, y_pred)
            score.append(r2)

            # The mean squared error
            print("Mean squared error: %.2f"
                  % mean_squared_error(powerY_test, y_pred))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(powerY_test, y_pred))

        axarr[funcIndex].plot(alphas, score)
        axarr[funcIndex].set_title(func)
        axarr[funcIndex].set_xlabel("alpha")
        axarr[funcIndex].set_ylabel("score")
        funcIndex = funcIndex + 1
        score = []

    plt.tight_layout()
    plt.show()


def compareOptFunctionViz():
    optimizationFunctions = ['lasso', 'lassolars', 'linearRegression', 'ridge', 'ridgecv']
    score = []

    for func in optimizationFunctions:
        if func == 'linearRegression':
            regr = linear_model.LinearRegression()
        elif func == 'ridge':
            regr = linear_model.Ridge(3)
        elif func == 'lasso':
            regr = linear_model.Lasso(0.003)
        elif func == 'lassolars':
            regr = linear_model.LassoLars(0.001)
        elif func == 'ridgecv':
            regr = linear_model.RidgeCV([10, 3, 2, 1, 0.1, 0.01, 0.001])

        # Train the model using the training sets
        regr.fit(powerX_train, powerY_train)

        # Make predictions using the testing set
        y_pred = regr.predict(powerX_test)
        r2 = r2_score(powerY_test, y_pred)
        score.append(r2)

    plt.plot(optimizationFunctions, score)
    plt.tight_layout()
    plt.show()

#############################

#linearRegression('lr')
#linearRegression('ridge')
#linearRegression('lasso')

#polinomialFunction()
#alphaViz()
compareOptFunctionViz()