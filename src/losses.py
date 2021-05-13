import numpy as np
from sklearn.metrics import mean_squared_error, pairwise_distances

class DistanceLoss():
    """
    Compute the distance loss.
    """
    
    def __init__(self):
        pass
    
    def _normalize(self, x):
        xmax, xmin = x.max(), x.min()
        x = (x - xmin)/(xmax - xmin)
        return x
    
    def __call__(self, X, Y):
        
        X = pairwise_distances(X)
        Y = pairwise_distances(Y)
        
        X = np.triu(X)
        Y = np.triu(Y)
        
        X = self._normalize(X)
        Y = self._normalize(Y)
        
        err = mean_squared_error(X, Y)
        
        return err