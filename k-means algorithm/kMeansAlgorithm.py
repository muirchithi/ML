import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import time as tm

#generate a random sample dataset for testing purposes
X, y = make_blobs(n_samples=1000,
                  random_state=0, cluster_std=0.60)

#format the random dataset to match a 2 dimensional csv-file input
formated_random_data = pd.DataFrame(X, columns=["parameter1","parameter2"])
x = formated_random_data[["parameter1", "parameter2"]]


#check if properly generated
print(x)
start = tm.time()

# visualize generated data points
plt.scatter(x["parameter2"],x["parameter1"], c="black")
plt.show()

#number of clusters
k=3

# Select random observation as centroids
centroids = (x.sample(n=k))
plt.scatter(x["parameter2"],x["parameter1"], c = "black")
plt.scatter(centroids["parameter2"], centroids["parameter1"], c = "red")
plt.show()



iterationDifference = 1
j = 0

while (iterationDifference>0.05):
    xd = x
    i = 1
    for index1, row_c in centroids.iterrows():
#creating an empty list to input all the computed length of all points
        ed =[]
#calculating the euclidean distance to calculate the distance all points have to the current centroids
        for index2,row_d in xd.iterrows():
            d1=(row_c["parameter2"]-row_d["parameter2"])**2
            d2=(row_c["parameter1"]-row_d["parameter1"])**2
            d=np.sqrt(d1+d2)
#append the distance of that given and calculated pint to the list
            ed.append(d)
#go into the next entry inside the list to not override the former result
        x[i]=ed
        i = i+1
    c = []
#check which distance is the smallest a data point has to all centroids
    for index,row in x.iterrows():
        print(row[1])
        min_distance =row[1]
        pos=1
        for i in range (k):
            if row[i+1] < min_distance:
                min_distance = row[i+1]
                pos= i+1
        c.append(pos)
    x["cluster"]=c
    newCentroids = x.groupby(["cluster"]).mean()[["parameter1","parameter2"]]

    if j == 0:
        iterationDifference=1
        j=j+1
    else:
        iterationDifference = (newCentroids["parameter1"] - centroids["parameter1"]).sum() + \
                              (newCentroids["parameter2"] - centroids["parameter2"]).sum()
        print(iterationDifference.sum())
    centroids = x.groupby(["cluster"]).mean()[["parameter1","parameter2"]]
end = tm.time()
#display the centroids with their assigned data points
color=["blue","green","cyan"]
for p in range(k):
    data=x[x["cluster"]==p+1]
    plt.scatter(data["parameter2"],data["parameter1"],c=color[p])
plt.scatter(centroids["parameter2"],centroids["parameter1"], c="red")
plt.show()

print("time in seconds is ")
print(end - start)
