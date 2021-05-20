import numpy as np
from sklearn.metrics import mean_squared_error, pairwise_distances
   
from sklearn.manifold._t_sne import _joint_probabilities, pairwise_distances
from scipy.spatial.distance import pdist

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
        err = 0

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

class TSNELoss():
    
    def __init__(self, parent_embeds, use_gaussians=True):
        
        # Cutoff to protect from division by zero errors
        self.MACHINE_EPSILON = np.finfo(np.double).eps
        
        # Hard code the perplexity param from TSNE
        self.perplexity = 30.0
        
        # Whether to use Gaussian PDFs for children
        self.use_gaussians = use_gaussians
        
        # Initialize loss funciton with parent embeddings
        self._load_joint_probs(parent_embeds)
    
    def _load_joint_probs(self, parent_embeds):
        """
        Load the loss function with the parent's joint probability 
        distribution over the pairwise distances.
        
        Parameters
        ----------
        parent_embeds : array-like of shape (n_samples, n_parent_neurons)
            Embeddings of the parent/base network
        """
        
        # Compute pairwise distance matrix
        distances = pairwise_distances(
            parent_embeds, 
            metric='euclidean', 
            squared=True
        )

        if np.any(distances < 0):
            raise ValueError("All distances should be positive, the "
                             "metric given is not correct")

        # Compute the joint probability distribution for the input space
        P = _joint_probabilities(distances, self.perplexity, verbose=0)
        
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")
        
        self.P = P

    def _child_dist(self, X_embedded, degrees_of_freedom, use_gaussians):
        """
        Get distribution over child layers' embeddings.

        Parameters
        ----------
        X_embedded : array-like of shape (n_samples, n_neurons)
            Embeddings of the child network's layer.
        degrees_of_freedom : int or float
            Degrees of freedom of the Student's t-distribution.
        use_gaussians : bool
            Compute distribution in child embeddings space using Gaussians.
            Else use Student's t-Distribution.
            
        Returns
        -------
        Q : array-like
            Distribution of child embeddings
        """
        
        if use_gaussians:
            dist = pairwise_distances(
                X_embedded, 
                metric='euclidean', 
                squared=True
            )
            Q = _joint_probabilities(dist, self.perplexity, 0)
        else:
            # Q is a heavy-tailed distribution: Student's t-distribution
            dist = pdist(X_embedded, "sqeuclidean")
            #return dist
            dist /= degrees_of_freedom
            dist += 1.
            dist **= (degrees_of_freedom + 1.0) / -2.0
            Q = np.maximum(dist / (2.0 * np.sum(dist)), self.MACHINE_EPSILON)
        
        return Q
        
    def _kl_div(self, Q):
        """
        Compute the KL divergence between distributions over the
        parent and child layers' embeddings.        
        
        Parameters
        ----------
        Q : array-like
            Distribution of child embeddings
        
        Returns
        -------
        kl_divergence : float
            Kullback-Leibler divergence of p_ij and q_ij.
        """

        # Optimization trick below: np.dot(x, y) is faster than
        # np.sum(x * y) because it calls BLAS

        # Objective: C (Kullback-Leibler divergence of P and Q)
        kl_divergence = 2.0 * np.dot(
            self.P, np.log(np.maximum(self.P, self.MACHINE_EPSILON) / Q))
            
        return kl_divergence

    def __call__(self, compressed_embeds):
        
        # Set the degrees of freedom for the Q distribution as
        # the number of neurons in the child's layer
        degrees_of_freedom = compressed_embeds.shape[1]
        
        Q = self._child_dist(
            compressed_embeds, 
            degrees_of_freedom, 
            self.use_gaussians
        )
        
        loss = self._kl_div(Q)
        
        return loss