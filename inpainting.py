import numpy as np

# The inpainting class defines an inverse problem where certain indices of the signal are missing.
class inpainting:
    """
    The inpainting problem is defined by its missing indices, which are the positions
    removed from the signal. This class sets up the likelihood for an inverse problem
    with the inpainting forward map.
    """
    def __init__(self, missing_index,sigma=0, data=0):
        """
        Initialize the inpainting problem.

        Args:
            missing_index (list or np.ndarray): Indices of the signal to be removed.
            sigma (float): Standard deviation of the likelihood noise. Default is 0.
            data (np.ndarray): Observed data. Default is 0.
        """
        
        self.index = missing_index # Indices to be removed
        self.data = data # Observed data for the likelihood
        self.lam = sigma # Standard deviation of the noise

        
    # Forward operation
    def forward(self,signal):
        """
        Apply the forward operation by removing the specified indices from the signal.

        Args:
            signal (np.ndarray): The input signal.

        Returns:
            np.ndarray: The signal with the specified indices removed.
        """
        
        # Deleting the indicies
        return np.delete(signal,self.index,0)

    
    def forward_adjoint(self,signal):
        """
        Compute the adjoint of the forward operation, which reconstructs the original
        signal by filling the missing indices with zeros.

        Args:
            signal (np.ndarray): The input signal.

        Returns:
            np.ndarray: The reconstructed signal with missing indices filled with zeros.
        """
        # Compute the total length of the original signal
        n = len(signal)+len(self.index)
        
        # Indices of the remaining elements in the original signal
        index_remain = np.delete(np.arange(0,n),self.index)

        # Initialize an array of zeros for the reconstructed signal
        a = np.zeros(n)

        # Assign the input signal to the remaining indices
        a[index_remain] = signal
        
        return a

    def set_data(self,signal, noise_level):
        """
        Generate synthetic data by applying the forward operation to the signal
        and adding additive Gaussian noise.

        Args:
            signal (np.ndarray): The ground truth signal.
            noise_level (float): Relative noise level as a percentage.
        """
        
        # Apply the forward operation to the signal
        Forward_data = self.forward(signal)

        # Compute the noise standard deviation
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data, ord=2))*1/np.sqrt(len(Forward_data))
        
        # Generate Gaussian noise
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        
        # Add noise to the forward data
        self.data = Forward_data + noise



    def jac_const(self,N):
        """
        Compute the forward operator as a matrix (Jacobian matrix).

        Args:
            N (int): The dimension of the input signal.

        Returns:
            np.ndarray: The forward operator matrix.
        """
        # Apply the forward operation to an identity matrix to compute the Jacobian
        return self.forward(np.identity(N))    
