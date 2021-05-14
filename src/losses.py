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
    
    def __call__(
        self, 
        X, # List of embeddings of the parent network
        Y, # List of embeddings of the child network
    ):
        
        assert len(X) == len(Y), \
            "X and Y have to be of equal length."

        # Accumulate the error of every embedding layer
        total_err = 0

        # Iterate through embeddings
        for i in range(len(X)):

            assert X[i].shape[0] == Y[i].shape[0],\
                "Both emeddings must be of equal size in first dimension."

            # Get pairwise distance matrix for each
            _x = pairwise_distances(X[i])
            _y = pairwise_distances(Y[i])
            
            # Zero lower triangle as matrix is symmetric
            _x = np.triu(_x)
            _y = np.triu(_y)
            
            # Normalize
            _x = self._normalize(_x)
            _y = self._normalize(_y)
            
            # Calculate MSE between distance matrices
            err += mean_squared_error(_x, _y)
        
        return err