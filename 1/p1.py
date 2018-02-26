import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas
import numpy
import random
import functions as func

dataX = {'x1': [], 'x2': [], 'y': []}
dataY = {'x1': [], 'x2': []}

numberOfRows = 500

# fill dictionary
for i in range(numberOfRows):
    x1, x2 = func.option7(i)
    dataX['x1'].append(x1)
    dataX['x2'].append(x2)
    dataX['y'].append(random.random())
    dataY['x1'].append(func.y(x1))
    dataY['x2'].append(func.y(x2))

# write first csv
ds1 = pandas.DataFrame(dataX)
ds1.to_csv("firstCSV")

# plot 2d graphs
f, axarr = plt.subplots(2)
axarr[0].set(xlabel='x')
axarr[0].set(ylabel='x1, x2')
axarr[0].plot(ds1['x1'], 'r', label='x1')
axarr[0].plot(ds1['x2'], 'g', label='x2')
axarr[1].set(xlabel='y(x1)')
axarr[1].set(ylabel='y(x2)')
axarr[1].plot(dataY['x1'], dataY['x2'], 'bo')
plt.tight_layout()  # graph 2 was covering x label of graph 1
plt.show()

# get averages
avgX1 = numpy.average(dataX['x1'])
avgX2 = numpy.average(dataX['x2'])
avgy = numpy.average(dataX['y'])
print("Average of x1: {}".format(avgX1))
print("Average of x2: {}".format(avgX2))
print("Average of y: {}".format(avgy))

# filter
newData = ds1[(ds1.x1 < avgX1) & (ds1.x2 < avgX2)]
del newData['y']

newData.to_csv("secondCSV")

# 3d plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(dataX['x1'], dataX['x2'])
plt.show()