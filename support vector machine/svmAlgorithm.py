import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


X, y = make_blobs (n_samples=2000, centers=2,
                   random_state=0, cluster_std=0.6)
formatted_data_set = pd.DataFrame(X, columns=["parameter1","parameter2"])
plt.scatter(formatted_data_set["parameter1"],formatted_data_set["parameter2"])
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = SVC(kernel="linear")
model.fit(X_train,y_train)

svc = LinearSVC()
svc.fit(X_train, y_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter');
ax = plt.gca()
xlim = ax.get_xlim()
w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - svc.intercept_[0] / w[1]
plt.plot(xx, yy)
yy = a * xx - (svc.intercept_[0] - 1) / w[1]
plt.plot(xx, yy, 'k--')
yy = a * xx - (svc.intercept_[0] + 1) / w[1]
plt.plot(xx, yy, 'k--')
plt.xlabel("n = 50000, cluster_std = 3.6, accuracy = 69.3")
plt.show()
print(model.score(X_test,y_test))

