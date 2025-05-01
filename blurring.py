import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft

# The blurring class represents a model for image blurring using a Gaussian kernel.
class blurring:
    def __init__(self,x, sigma_kernel,Factor=1 ,lam=0, data=0):
                """
        Initialize the blurring model.

        Args:
            x (np.ndarray): Grid points or spatial domain for the kernel.
            sigma_kernel (float): Standard deviation of the Gaussian kernel.
            Factor (int): Downsampling factor for the signal. Default is 1.
            lam (float): Noise level. Default is 0.
            data (np.ndarray): Observed data. Default is 0.
        """
        
        self.grid=x # Spatial grid
        self.sigma = sigma_kernel # Standard deviation of the Gaussian kernel
        self.data = data # Observed data
        self.lam = lam # Noise level
        self.Factor = Factor # Downsampling factor

    def kernel(self):
         """
        Generate the Gaussian kernel for blurring.

        Returns:
            np.ndarray: The normalized Gaussian kernel.

        Raises:
            ValueError: If sigma is not within the range (0, 1/6].
        """
        
        sigma = self.sigma

        # Check if sigma is within the valid range
        if np.logical_and(0 < sigma, sigma <= 1 / 6):
            # Continuous normalization constant ensuring unit integration of the kernel
            c_sigma = 1 / (2.5 * sigma)
            
            # Define the Gaussian kernel over the grid
            c= np.piecewise(self.grid,
                            [self.grid <= 3 * sigma, np.logical_and(3 * sigma <= self.grid, self.grid <= 1 - 3 * sigma),
                            np.logical_and(1 - 3 * sigma < self.grid, self.grid <= 1)],
                            [lambda x: c_sigma * np.exp(-x ** 2 / (2 * sigma ** 2)), lambda x: 0,
                            lambda x: c_sigma * np.exp(-(x - 1) ** 2 / (2 * sigma ** 2))])
            # Normalize the kernel to ensure unit sum
            return  c/ np.linalg.norm(c, ord=1)
        else:
            print("Sigma value is not supported")
            exit()
        

    def forward(self,signal):
        """
        Apply the forward blurring operation on a signal.

        Args:
            signal (np.ndarray): The input signal to be blurred.

        Returns:
            np.ndarray: The blurred and downsampled signal.
        """
        # Ensure the kernel and signal have the same length
        if len(self.kernel()) == len(signal):
            # Perform convolution in the frequency domain
            result = ifft(fft(self.kernel())*fft(signal))
            
        # Downsample the result based on the specified factor   
        return result.real[0::self.Factor]

    def set_data(self,signal, noise_level):
        """
        Generate noisy observed data by applying the forward model 
        to the signal and adding Gaussian noise.

        Args:
            signal (np.ndarray): The input signal.
            noise_level (float): Relative noise level as a percentage.
        """
        
        # Apply the forward model to the signal
        Forward_data = self.forward(signal)

        # Compute the noise standard deviation
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data, ord=2))/np.sqrt(len(Forward_data))
        
        # Generate Gaussian noise
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        
        # Add noise to the forward data
        self.data = Forward_data + noise


    def jac_const(self,N):
        """
        Compute the Jacobian matrix (or linear operator) for the blurring operation.

        Args:
            N (int): Dimension of the signal.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        # Create an identity matrix
        I = np.identity(N)

        # Apply the forward model to each column of the identity matrix
        return self.forward(I)    

