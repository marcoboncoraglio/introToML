# from keras import models
# from keras import layers

import numpy as np

from sklearn.linear_model import perceptron
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
@attribute having_IP_Address  { -1,1 }
@attribute URL_Length   { 1,0,-1 }
@attribute Shortining_Service { 1,-1 }
@attribute having_At_Symbol   { 1,-1 }
@attribute double_slash_redirecting { -1,1 }
@attribute Prefix_Suffix  { -1,1 }
@attribute having_Sub_Domain  { -1,0,1 }
@attribute SSLfinal_State  { -1,1,0 }
@attribute Domain_registeration_length { -1,1 }
@attribute Favicon { 1,-1 }
@attribute port { 1,-1 }
@attribute HTTPS_token { -1,1 }
@attribute Request_URL  { 1,-1 }
@attribute URL_of_Anchor { -1,0,1 }
@attribute Links_in_tags { 1,-1,0 }
@attribute SFH  { -1,1,0 }
@attribute Submitting_to_email { -1,1 }
@attribute Abnormal_URL { -1,1 }
@attribute Redirect  { 0,1 }
@attribute on_mouseover  { 1,-1 }
@attribute RightClick  { 1,-1 }
@attribute popUpWidnow  { 1,-1 }
@attribute Iframe { 1,-1 }
@attribute age_of_domain  { -1,1 }
@attribute DNSRecord   { -1,1 }
@attribute web_traffic  { -1,0,1 }
@attribute Page_Rank { -1,1 }
@attribute Google_Index { 1,-1 }
@attribute Links_pointing_to_page { 1,0,-1 }
@attribute Statistical_report { -1,1 }
@attribute Result  { -1,1 }
"""

df = pd.read_csv("phishing_websites.txt")
# print(df.head())
y = df['Result']
x = df.drop(['Result'], axis=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

df_train = df[:-3316]
dfs_test = df[-3316:]

# Create the perceptron object (net)
net = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

# Train the perceptron object (net)
net.fit(df_train[['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
                'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon',
                'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
                'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
                'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report']
                ], df_train['Result'])

# Output the coefficints
print("Coefficient 0: " + str(net.coef_[0, 0]))
print("Coefficient 1: " + str(net.coef_[0, 1]))
print("Bias " + str(net.intercept_))

# Do a prediction
pred = net.predict(df[['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
                           'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
                           'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
                           'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect',
                           'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord',
                           'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report']])
#prediction
print(pred)

# Confusion Matrix
confusion_matrix(pred, df['Result'])

"""

learning_rates = [0.01, 0.04, ...]
alphas = [0.01]
optimizers = ['adam','lbfgs', ...]
for plt in optimizers:
    for lr in learning_rates:
        for alpha in alphas:
            mlp = MLPClassfier(hidden_layer=[],alpha=alpha[], learning_rate = lr, solver=opt)
            mlp.fit()
            make scoriing of model(f1 or r2 or accuracy or ...)
    build 3d plot loss(learning_rates, alphas)

"""


# MLPClassifier
print("\nMLPClassifier: ")
y = df['Result']
x = df.drop(['Result'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

learning_rates = ['constant', 'invscaling', 'adaptive']
alphas = [0.0001, 0.0001, 0.01, 0.04, 0.08, 0.16, 0.32, 0.64]
optimizers = ['adam', 'lbfgs', 'sgd']

for opt in optimizers:
    for lr in learning_rates:
        for alpha in alphas:
            mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), verbose=True, max_iter=50, alpha=alpha,
                                learning_rate=lr, solver=opt, tol=0.00001)
            mlp.fit(x_train, y_train)

            # make scoring of model(f1 or r2 or accuracy or ...)
            y_pred = mlp.predict(x_test)
            acc_score = accuracy_score(y_test, y_pred)
            print("Accuracy Score: ", acc_score)

    # build 3d plot loss(learning_rates, alphas)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(dataX['x1'], dataX['x2'])

# clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
#                     solver='sgd', verbose=10,  random_state=21, tol=0.000000001)

# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)


print("Accuracy Score:", acc_score)
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, center=True)
plt.show()

"""
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
"""
