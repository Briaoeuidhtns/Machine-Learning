"""A library of machine learning tools.

Snippets of project descriptions stolen as doctests

Our first task will be to add the perceptron part. We will use pandas, numpy
and mathplotlib, so let us import those at once.

>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt

Need an output directory for plots since not running interactively, so let's
create one:

>>> from pathlib import Path
>>> plot_folder = Path('../plots')
>>> for f in plot_folder.glob('*.png'):
...     f.unlink()
>>> plot_folder.mkdir(exist_ok=True)

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

Let us visualize the data, to make sure that it is separable:
Using `_=` to ignore output in doctest, since it doesn't matter here.

>>> _=plt.figure()
>>> _=plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
>>> _=plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
>>> _=plt.xlabel('petal length')
>>> _=plt.ylabel('sepal length')
>>> _=plt.legend(loc='upper left')
>>> _=plt.savefig(plot_folder/'separable.png')

Our goal is to be able to do the following:

I am initializing class `Perceptron` from `ml.py`. The first argument is the
learning rate and this is a number between 0 and 1.

>>> pn = Perceptron(0.1, 1000)

After we run the fit-function on the perceptron, we can extract (so we are
able to plot) the misclassification errors:

I trimmed the error result from the example, and used a np array. The test
output has been modified to test that.

>>> pn.fit(X, y)
>>> pn.errors
array([2, 2, 3, 2, 1, 0])

Here I fitted X to y i.e. the algorithm found the appropriate values for the
weights (w). I also printed the error for each iteration.

>>> _=plt.figure()
>>> _=plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
>>> _=plt.xlabel('Iteration')
>>> _=plt.ylabel('# of misclassifications')
>>> _=plt.savefig(plot_folder/'errors.png')

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
array([-0.4 , -0.68, 1.82])

As this is a 2D example we should be able to visualize this boundary. There is
no real good function in Matplotlib to do this, so we are going to create a
helper function and put that in our ML.py library, `plot_decision_regions`.

>>> plot_decision_regions(X, y, pn)
>>> plt.savefig(plot_folder/'boundary.png')

The other data I tested against:

Really close data. 701 seems like a really big number, but they are close so
I'm assuming it could be right.

>>> X1 = df.iloc[0:100, [0, 1]].values
>>> pn.fit(X1, y)
>>> pn.errors.shape
(701,)
>>> plot_decision_regions(X1, y, pn)
>>> plt.savefig(plot_folder/'close_boundary.png')

>>> X2 = df.iloc[0:100, [2, 3]].values
>>> pn.fit(X2, y)
>>> pn.errors
array([2, 2, 0])
>>> plot_decision_regions(X2, y, pn)
>>> plt.savefig(plot_folder/'big_boundary.png')

Close all the figures we opened:

>>> plt.close('all')
"""
from __future__ import annotations

from typing import Optional

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
    # Create clean figure
    plt.figure()

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

    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend()


class Perceptron:
    def __init__(self, rate=0.01, niter=0):
        if not 0 <= rate <= 1:
            raise ValueError('Rate must be between 0 and 1')
        if niter < 0:
            raise ValueError('Number of iterations must be positive')

        self.__rate = rate
        self.__niter = niter

        self.__errors: Optional[np.ndarray] = None
        self.__weights: Optional[np.ndarray] = None

    @property
    def errors(self):
        if self.__errors is not None:
            err = self.__errors[:]
            err.flags.writeable = False
            return err

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit training data
           X : Training vectors, X.shape : [#samples, #features]
           y : Target values, y.shape : [#samples]
        """
        samples, features = X.shape

        if not y.shape == (samples,):
            raise ValueError(
                f'Sample sizes must match, got X: {X.shape}, y: {y.shape}')

        # weights: create a weights array of right size
        # and initialize elements to zero
        self.__weights = np.zeros(features + 1)

        # Number of misclassifications, creates an array
        # to hold the number of misclassifications
        self.__errors = np.zeros(self.niter, dtype=int)

        # main loop to fit the data to the labels
        for i in range(self.niter):
            # loop over all the objects in X and corresponding y element
            for x_i, target in zip(X, y):
                # calculate the needed (delta_w) update from previous step
                # delta_w = rate * (target – prediction current object)
                delta_w = self.rate * (target - self.predict(x_i))

                # calculate what the current object will add to the weight
                self.__weights[1:] += delta_w * x_i

                # set the bias to be the current delta_w
                self.__weights[0] += delta_w

                # increase the iteration error if delta_w != 0
                if delta_w != 0:
                    self.__errors[i] += 1

            if self.__errors[i] == 0:
                self.__errors = self.__errors[:i+1]
                break
        else:
            raise RuntimeError(
                f'Did not converge within {self.niter} iterations')

    def net_input(self, X):
        """Calculate net input"""
        # return the return the dot product: X.w + bias
        return X.dot(self.__weights[1:]) + self.__weights[0]

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
        if self.__weights is not None:
            w = self.__weights[:]
            w.flags.writeable = False
            return w


def _test():
    import doctest
    return doctest.testmod(optionflags=0
                           | doctest.ELLIPSIS
                           | doctest.NORMALIZE_WHITESPACE)


if __name__ == '__main__':
    failures, tests = _test()
    exit(failures)

# ignore below, random testing things
if __name__ == '__repl__':
    print(_test())

if __name__ == '__live_coding__':
    import pandas as pd
    df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        **IRIS_OPTIONS)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    pn = Perceptron(0.1, 10)
    pn.fit(X, y)
    print(pn.errors)
    print(pn.net_input(X))
    print(pn.predict(X))
    print(pn.weight)
    plot_decision_regions(X, y, pn)
    plt.savefig('plot.png')
