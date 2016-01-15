#########################################
# Berkeley Carpool Project

# Constructed a Mean Shift algorithm and used hierarchical clustering to partition longitude and latitude data points into groups 
# Used to identify high-density locations that users travel to most frequently

import numpy as np
from sklearn.cluster import MeanShift 
from sklearn import preprocessing, datasets
import matplotlib.pyplot as plt 
import csv

# read longitude and latitude values from file
f = open('Kmeans_data.csv')
csv_f = csv.reader(f)

X = []
for row in csv_f:
	row[0], row[1] = float(row[0]), float(row[1])
	X.append(row)
X = np.array(X)

# plot location points
plt.scatter(X[:,0],X[:,1],s=60)
plt.show()

# perform k-means algorithm on data, identify centroids
ms = MeanShift()
ms.fit(X)

labels = ms.labels_
centroids = ms.cluster_centers_
print("Central locations: ", centroids)

n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters_)

# color each cluster and plot each with corresponding centroid
colors = 10*['r.','g.','c.','k.','y.','m.']

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize =13)

plt.scatter(centroids[:,0],centroids[:,1],
	marker="x", s=60, linewidths = 3, zorder=10)
plt.show()




