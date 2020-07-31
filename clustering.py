import pandas

df= pandas.read_csv('forestfires.csv')
print(df)

subset = df[['temp', 'wind', "area"]]
array = subset.values
X = array[:, 0:3]
# we have no target y, hence it is unsupervised

from sklearn.cluster import KMeans
# elbow plot
# justpaste.it/26p9o

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
from sklearn.cluster import KMeans
wcss = [] # wcss = within cluster sum of squares
for i in range(1, 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# from elbow
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X) # train unsupervised way

# get the center number(mean0 of each cluster, for our 3 features)
centronoids = kmeans.cluster_centers_
dataframe = pandas.DataFrame(centronoids, columns=['temp', 'wind', "area"])
print(dataframe)

# access the data for the clusters
result = zip(X, kmeans.labels_) # kmeans label (group label)
sorted_results = sorted(result, key=lambda x:x[1])
print(sorted_results)

for result in sorted_results:
     print(result)