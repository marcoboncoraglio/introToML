from sklearn.linear_model import perceptron
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

perc_alphas = [0.00001, 0.00005, 0.0001, 0.0003, 0.0005, 0.0008, 0.001]
perc_acc_score = []
for alpha in perc_alphas:
    # Create the perceptron object (net)
    net = perceptron.Perceptron(alpha=alpha, max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

    # Train the perceptron object (net)
    net.fit(x_train, y_train)
    print("alpha: ", alpha)

    # Do a prediction
    pred = net.predict(x_test)
    acc = accuracy_score(y_test, pred) * 100
    print("accuracy: {0:2f}%".format(acc))
    perc_acc_score.append(acc)
    # print classification report
    print(classification_report(y_test, pred))


# create 2d graph alpha by accuracy
plt.xlabel("alpha")
plt.ylabel("accuracy score")
plt.plot(perc_alphas, perc_acc_score)
plt.show()

# The graph show us with alpha = 0.0003 an accuracy of 91.678726% would be reached
# Create the perceptron object (net)
net = perceptron.Perceptron(alpha=0.0003, max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
# Train the perceptron object (net)
net.fit(x_train, y_train)
# Output the coefficints
for i in range(len(x.columns)):
    print("Coefficient " + str(i + 1) + ": " + str(net.coef_[0, i]))
print("Bias " + str(net.intercept_))
# Do a prediction
pred = net.predict(x_test)
acc = accuracy_score(y_test, pred) * 100
print("accuracy: {0:2f}%".format(acc))
# print classification report?
print(classification_report(y_test, pred))

# print confusion matrix using seaborn
cm = confusion_matrix(y_test, pred)
print("Coefficient Matrix: ")
print(cm)
sns.heatmap(cm, center=True)
plt.show()

# MLPClassifier
print("\nMLPClassifier: ")
y = df['Result']
x = df.drop(['Result'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

learning_rates = ['constant', 'invscaling', 'adaptive']
alphas = [0.0001, 0.001, 0.01]
optimizers = ['adam', 'lbfgs', 'sgd']
alpha_list = []
acc_list = []

for opt in optimizers:
    print("optimizer: ", opt)
    for lr in range(len(learning_rates)):
        print("learning rate:", learning_rates[lr])
        for alpha in alphas:
            print("alpha: ", alpha)
            mlp = MLPClassifier(max_iter=50, alpha=alpha,
                                learning_rate=learning_rates[lr], solver=opt, tol=0.00001)
            mlp.fit(x_train, y_train)
            y_pred = mlp.predict(x_test)
            acc_score = accuracy_score(y_test, y_pred)
            acc_list.append(acc_score)
            alpha_list.append(alpha)

            # TODO: print classification report?
            # print(classification_report(y, y_pred))

            print("Accuracy Score: ", acc_score)

# build 3d plot loss(learning_rates, alphas)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title(optimizers[0])
ax.set_xlabel("alpha")
ax.set_ylabel("0=constant, 1=invscaling, 2=adaptive")
ax.set_zlabel("acc %")
ax.plot(alpha_list[:3], range(3), acc_list[:3])
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title(optimizers[1])
ax.set_xlabel("alpha")
ax.set_ylabel("0=constant, 1=invscaling, 2=adaptive")
ax.set_zlabel("acc %")
ax.plot(alpha_list[3:6], range(3), acc_list[3:6])
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("alpha")
ax.set_ylabel("0=constant, 1=invscaling, 2=adaptive")
ax.set_zlabel("acc %")
ax.set_title(optimizers[2])
ax.plot(alpha_list[6:9], range(3), acc_list[6:9])
plt.show()
