import matplotlib.pyplot as plt

from sklearn.linear_model import perceptron
from pandas import *
from sklearn.metrics import accuracy_score

inputs = pandas.read_csv('./phishing_websites.txt')
inputs_train = inputs[:-3316]
inputs_test = inputs[-3316:]

# Create the perceptron object (net)
net = perceptron.Perceptron(max_iter=1000, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

# Train the perceptron object (net)
features = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
             'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port',
             'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
             'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
             'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report']

net.fit(inputs_train[features], inputs_train['Result'])

print(len(features))
# Output the coefficints
for i in range(len(features)):
    print("Coefficient " + str(i+1) + ": " + str(net.coef_[0,i]))

print("Bias " + str(net.intercept_))

# Do a prediction

inputs_test_list = inputs_test[features]

pred = net.predict(inputs_test_list)

# Confusion Matrix
# confusion_matrix(pred, inputs['Result'])
print("accuracy: {0:2f}%".format(accuracy_score(inputs_test[['Result']], pred) *100))