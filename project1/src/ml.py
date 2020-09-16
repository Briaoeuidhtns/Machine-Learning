"""A library of machine learning tools.

Snippets of project descriptions stolen as doctests

Our first task will be to add the perceptron part. We will use pandas, numpy and
mathplotlib, so let us import those at once.

>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt

So our first task will be to extract the 100 first class labels.

>>> df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', **IRIS_OPTIONS)
>>> y = df.iloc[0:100, 4].values
>>> y
array(['Iris-setosa', ..., 'Iris-versicolor'], dtype=object)

The Perceptron model we studied used Y = {-1, 1} so we better convert our label
to those integer class labels

>>> y = np.where(y == 'Iris-setosa', -1, 1)
>>> y
array([-1, -1, -1, ..., 1, 1, 1])

Next we need to extract the features. We could use all four features, but as I
mentioned the classes has to be separable, which meant we had to be able to cut
the data into two clean groups with a hyperplane. So we are just going to use
two features sepal length (the first feature) and petal length (the third
feature).

>>> X = df.iloc[0:100, [0, 2]].values
>>> X
array([...])

{Omitted visualizations}

Our goal is to be able to do the following:

I am initializing class `Perceptron` from `ml.py`. The first argument is the
learning rate and this is a number between 0 and 1.

>>> pn = Perceptron(0.1, 10)

After we run the fit-function on the perceptron, we can extract (so we are
able to plot) the misclassification errors:

>>> pn.fit(X, y)
>>> pn.errors
[2, 2, 3, 2, 1, 0, 0, 0, 0, 0]

Here I fitted X to y i.e. the algorithm found the appropriate values for the
weights (w). I also printed the error for each iteration.

{Omitted visualizations}

The perceptron converged after 6 iterations, and from there and on we should
have been able to classify all training samples with a zero error rate.

Time to look what I have in the perceptron class:

>>> pn.net_input(X)
array([-1.32 , -1.184, ..., 1.592, 3.186])
>>> pn.predict(X)
array([-1, -1, ..., 1, 1])

As I said earlier, our weight vector creates a hyper-plane or a decision
boundary between our two classes. We can look at our w-array
>>> pn.weight
array([-0.4, -0.68, 1.82])

"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

IRIS_OPTIONS = {
    'header': None,
    'names': (
        'SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm',
        'Species',
    ),
}


def plot_decision_regions(X, y, classifier: Perceptron, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


class Perceptron:
    def __init__(self, rate=0.01, niter=10):
        if not 0 <= rate <= 1:
            raise ValueError('Rate must be between 0 and 1')
        if niter < 0:
            raise ValueError('Number of iterations must be positive')

        self.__rate = rate
        self.__niter = niter

    @property
    def errors(self):
        raise NotImplementedError()

    def fit(self, X, y):
        """Fit training data
           X : Training vectors, X.shape : [#samples, #features]
           y : Target values, y.shape : [#samples]
        """

        # weights: create a weights array of right size
        # and initialize elements to zero

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # set iteration error to zero
            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                # calculate the needed (delta_w) update from previous step
                # delta_w = rate * (target – prediction current object)

                # calculate what the current object will add to the weight

                # set the bias to be the current delta_w

                # increase the iteration error if delta_w != 0
                ...

            # Update the misclassification array with # of errors in iteration

        raise NotImplementedError()

    def net_input(self, X):
        """Calculate net input"""
        # return the return the dot product: X.w + bias
        raise NotImplementedError()

    @property
    def niter(self):
        return self.__niter

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    @property
    def rate(self):
        return self.__rate

    @property
    def weight(self):
        raise NotImplementedError()


def _test(**kwargs):
    import doctest
    return doctest.testmod(**kwargs)


if __name__ in {'__main__', '__live_coding__'}:
    failures, tests = _test()
    exit(failures)
