from itertools import cycle

import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation, estimate_bandwidth, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score, v_measure_score, completeness_score, fowlkes_mallows_score
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

dataset = pd.read_csv("Wholesale customers data.csv")

X = np.array(dataset.drop(['Region'], axis=1).astype(float))
X = StandardScaler().fit_transform(X)
y = np.array(dataset['Region'])

res_homo = []
res_complete = []
res_v = []
res_fowl = []

# divide in three clusters
clf = KMeans(n_clusters=3)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

# make components visualizable in 3d
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)

# one color for each region
color = ['r', 'g', 'b']

# plot regions
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("Regions in function of pca (3) of all other columns")

for i in range(len(dataset)):
    ax.scatter(principalComponents[i][0], principalComponents[i][1], principalComponents[i][2], c=color[y[i] - 1])
plt.show()

# plot clusters
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("K-Means clusters")

for i in range(len(dataset)):
    ax.scatter(principalComponents[i][0], principalComponents[i][1], principalComponents[i][2], c=color[labels[i]])
plt.show()

hs = homogeneity_score(labels, y)
complete = completeness_score(labels, y)
v = v_measure_score(labels, y)
f = fowlkes_mallows_score(labels, y)

res_homo.append(hs)
res_complete.append(complete)
res_v.append(v)
res_fowl.append(f)

print("K-means homogeneity: " + str(hs))
print("K-means completeness: " + str(complete))
print("K-means v score: " + str(v))
print("K-means fowlkes_mallows: " + str(f))


###########################################################################

af = AffinityPropagation().fit(X)
color = ['r', 'g', 'b']

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Affinity Prop: Estimated number of clusters: %d' % n_clusters_)
plt.show()

hs = homogeneity_score(labels, y)
complete = completeness_score(labels, y)
v = v_measure_score(labels, y)
f = fowlkes_mallows_score(labels, y)

res_homo.append(hs)
res_complete.append(complete)
res_v.append(v)
res_fowl.append(f)

print("\nAF homogeneity: " + str(hs))
print("AF completeness: " + str(complete))
print("AF v score: " + str(v))
print("AF fowlkes_mallows: " + str(f))

############################################################################

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('MeanShift: Estimated number of clusters: %d' % n_clusters_)
plt.show()

hs = homogeneity_score(labels, y)
complete = completeness_score(labels, y)
v = v_measure_score(labels, y)
f = fowlkes_mallows_score(labels, y)

res_homo.append(hs)
res_complete.append(complete)
res_v.append(v)
res_fowl.append(f)

print("\nMS homogeneity: " + str(hs))
print("MS completeness: " + str(complete))
print("MS v score: " + str(v))
print("MS fowlkes_mallows: " + str(f))

#####################################################################

algs = ['kmeans', 'af', 'ms']

plt.scatter(algs,res_homo)
plt.title('Homogeneity score')
plt.show()

plt.scatter(algs,res_complete)
plt.title('completeness score')
plt.show()

plt.scatter(algs,res_v)
plt.title('v score')
plt.show()

plt.scatter(algs,res_fowl)
plt.title('fowlkes mallows score')
plt.show()