import numpy as np;

class LinearClassificationModel(object):

    def __init__(self):
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

        if self.W == None:
            self.W = 0.001 * np.random.rand(dim, num_class) #initial W as (N, C)

        loss_history = []
        for iter in range(num_iters):
            X_batch = None
            y_batch = None

            #sample from X with batch_size
            batch_index = np.random.choice(num_train, batch_size, False)
            X_batch = X[batch_size]
            y_batch = y[batch_size]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

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
        m = X.shape[0]
        predictions = np.zeros(m)

        scores = np.dot(X, self.W)
        predictions = np.argmax(scores, axis=1)

        return predictions

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