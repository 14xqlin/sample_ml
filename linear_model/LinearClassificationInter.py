import numpy as np;

from CrossEntropy import *

class LinearClassification(object):

    def __init__(self):
        self.n_dims = None
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Training classifier with mini batch optimization

        Inputs:
        - X: numpy array about datas, with shape(m, n); m train datas with n dimensions.
        - y: numpy array about labels, with shape(m,); y[i] = c, 0 <= c < C for C classes.
        - learning_rate: (float) learning rate about optimization
        - reg: (float) regularization strength
        - num_iters: (int) number of steps about optimization
        - batch_size: (int) number of train samples for each step
        - verbose: (boolean) If true, print process during optimization

        Return:
        A list about loss value for each step in iterations
        """
        num_train, dim = X.shape
        num_class = np.max(y) + 1

        loss_history = []
        self.initW(self.n_dims)

        for iter in range(num_iters):
            X_batch = None
            y_batch = None

            #sample from X with batch_size
            batch_index = np.random.choice(num_train, batch_size, False)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            loss, grad = self.loss(X_batch, y_batch, reg)

            #update W
            self.W = self.W - learning_rate * grad

            if (verbose) and (iter % 100 == 0):
                print( 'iteration %d / %d: loss: %f' % (iter, num_iters, loss) )

        return loss_history

    def predict(self, X):
        """

        Inputs:
        - X: numpy array about datas, with shape(m, n); m train datas with n dimensions.

        Returns:
            predictions: predicted labels about X, with shape(m,)
        """

    def loss(self, X_batch, y_batch, reg):
        """
        compute loss and its derivative.

        Inputs:
        - X_batch: numpy array about training datas, with shape(m, n)
        - y_batch: numpy array about training datas' labels, with shape(m,)
        - reg: (float) regularization

        Returns:
        - loss: (float) loss value
        - grad: numpy array about self.W derivative, with the shape as the same as self.W
        """

    def initW(self, n_dims):
        """
        :param n_dims:
        :return:
        """

class Logistic(LinearClassification):

    def __init__(self, n_dims):
        super(Logistic, self).__init__()
        # self.W = 0.001 * np.random.rand(n_dims)
        self.n_dims = n_dims
        self.initW(self.n_dims)
        # self.W = np.zeros(n_dims)

    def loss(self, X_batch, y_batch, reg):
        return logistic_loss(self.W, X_batch, y_batch, reg)

    def initW(self, n_dims):
        self.W = np.zeros(n_dims)

    def predict(self, X):

        scores = np.dot(X, self.W)
        scores[ np.where(scores >= 0)[0] ] = 1
        scores[np.where(scores < 0)[0]] = 0

        return scores
