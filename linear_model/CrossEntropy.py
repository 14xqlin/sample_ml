import numpy as np

def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x) )

def logistic_loss(W, X, y, reg):
    """
    Inputs
    - W: weight aobut model, of shape(n, )
    - X: numpy array about training datas, of shape(m, n)
    - y: numpy array about training labels, of shape(m,)
    - reg: (float), regularization strength

    Return:
    - loss: (float), loss value
    - dW: numpy array with the same shape as W
    """
    loss = 0.0
    m, n = X.shape

    scores = np.dot(X, W) #shape(m,)
    p = sigmoid(scores)
    loss = np.sum( -1.0 * (y * np.log(p) + (1 - y) * np.log(1 - p)) ) / m + \
           0.5 * reg * np.sum(W ** 2)

    dW = np.dot( X.T, p - y)
    dW /= m
    dW += reg * W

    return loss, dW