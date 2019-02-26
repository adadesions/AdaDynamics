import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def xy2polar(x, y):
    r = np.sqrt(np.power(x, 2)+np.power(y, 2))
    try:
        theta = np.arctan(y/x)*57.296
    except ValueError:
        theta = 0

    return [r, theta]


def data2polar(data):
    result = []
    for d in data:
        (x, y) = d
        polar = xy2polar(x, y)
        result.append(polar)

    return np.array(result)


iris = datasets.load_iris()
xf, yf = 2, 3
x = iris.data[:, (xf, yf)]
y = (iris.target == 2).astype(np.float64)

svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svm', LinearSVC(C=1, loss='hinge'))
])

iris_polar = data2polar(iris.data[:, (xf, yf)])

svm_clf.fit(x, y)
predicted = svm_clf.predict([[5.5, 1.7]])
print(predicted)

formatter = plt.FuncFormatter(lambda i, *arg: iris.target_names[int(i)])
fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection='polar')
# ax.scatter(iris_polar[:, 0], iris_polar[:, 1], c=iris.target)
# ax.plot(iris_polar[:, 1], iris_polar[:, 0])
plt.scatter(iris.data[:, xf], iris.data[:, yf], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[xf])
plt.ylabel(iris.feature_names[yf])

plt.tight_layout()
plt.show()