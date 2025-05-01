import numpy as np
import pywt # Python Wavelet Transform library
from scipy.stats import gennorm, norm # Generalized Normal and Normal distributions
from scipy import sparse # Sparse matrix operations
import matplotlib.pyplot as plt
from scipy.special import gamma # Gamma function

# Class that defines the Besov prior for 2D problems with all its functionalities
class besov_prior2D:
    def __init__(self,J,delt,level, slices, shape ,s=1, p=1, wavelet="db1"):
        """
        Initialize the 2D Besov prior.

        Args:
            J (int): Maximum number of wavelet levels (dimension is 2^(2*J)).
            delt (float): Regularization parameter for the prior.
            level (int): Levels to exclude from the Discrete Wavelet Transform.
            slices (list): Structure of the 2D wavelet coefficients for transformation.
            shape (list): Shape of the wavelet coefficients.
            s (float): Smoothness parameter of the Besov space. Default is 1.
            p (float): Regularity parameter of the Besov space. Default is 1.
            wavelet (str): Wavelet basis to use for the transformation. Default is "db1".
        """
        self.s = s # Smoothness parameter
        self.p = p # Regularity parameter
        self.wavelet = wavelet # Wavelet basis
        self.J = J # Total levels in wavelet decomposition
        self.delt = delt  # Regularization parameter
        self.level = level # Levels to exclude
        self.coeff_slices = slices # 2D wavelet coefficient structure
        self.coeff_shape = shape # Shape of wavelet coefficients

    def sample(self):
        """
        Generate a random sample from the Besov prior.

        Returns:
            np.ndarray: A sample generated from the Besov prior.
        """
        # Compute the scale parameter for the generalized normal distribution
        scale = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)

        # Sample coefficients from the generalized normal distribution
        xi = gennorm.rvs(self.p, scale=scale, size=2**(self.J*2))
        
        # Compute the inverse Besov weights
        weights = self.inverse_weights()
        
        # Compute the scaled coefficients
        coefficients = (weights/np.linalg.norm(weights,ord=2))*xi
        
        # Perform the inverse wavelet transform to generate the sample
        return pywt.waverec2(pywt.unravel_coeffs(coefficients, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')


    def inverse_weights(self):
        """
        Compute the inverse wavelet weights for the Besov prior.

        Returns:
            np.ndarray: Array of inverse weights.
        """
        
        # Total dimension
        n = 2**(self.J*2)
        
        # Allocate space for weights
        inv_weights = np.zeros(n)
        
        # Compute weights for the scaling function
        inv_weights[0:2**((self.level)*2)] = 2**(-self.level*(self.s+1.0-2.0/self.p))
        
        # Compute weights for each wavelet level
        for i in range(self.level,self.J):
            inv_weights[2**(i*2):2**((i+1)*2)]=2**(-i*(self.s+1.0-2.0/self.p))
            
        return inv_weights



    def weights(self):
        """
        Compute the wavelet weights for the Besov prior.

        Returns:
            np.ndarray: Array of weights.
        """
        # Total dimension
        n = 2**(self.J*2)

        # Initializing array for the weights
        weights = np.zeros(n)

        # Scaling coefficients weight
        weights[0] = 1

        # Compute weights for each wavelet level
        for i in range(self.J):
            weights[2**(i*2):2**((i+1)*2)]=2**(i*(self.s+1.0-2.0/self.p))
            
        return weights          
    
    def inv_wavelet_weigth(self, signal):
        """
        Apply the inverse wavelet transform with inverse Besov weights.

        Args:
            signal (np.ndarray): Input signal to be transformed.

        Returns:
            np.ndarray: Signal transformed using inverse Besov weights.
        """
        
        # Compute the inverse Besov weights
        inv_weigth = self.inverse_weights()

        # Normalization of coeffcients
        size = np.linalg.norm(inv_weigth,ord=2) # Normalizing factor
        coeff = (inv_weigth/size)*signal # Normalization of scaling coefficients
        coeff[0:2**(self.level*2)]=1/size*signal[0:2**(self.level*2)] # Normalization of wavelet coefficients

        # Perform the inverse wavelet transform
        return pywt.waverec2(pywt.unravel_coeffs(coeff, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')
    

    
    def wavelet_weigth(self, signal):
        """
        Apply the wavelet weights to the input signal.

        Args:
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Weighted wavelet coefficients.
        """
        # Computing Besov weights
         weights = self.weights()

        # Computing wavelet coefficients of the signal
         wavelet_coefficients = pywt.wavedec2(signal.reshape((2**self.J,2**self.J)), self.wavelet, mode='periodization', level=self.J-self.level)

        # Normalizing the weights
         size = np.linalg.norm(weights**-1,ord=2) # Normalization constant
         weights[0:2**(self.level*2)] = size # Normalizing scaling coefficient weights
         weights[2**(self.level*2)::]*= size # Normalizing wavelet coefficient weights

        # Return the weighted wavelet coeffcients as an array
         return weights*pywt.ravel_coeffs(wavelet_coefficients)[0]
    
    def wavelet_weight_adjoint(self,signal):
        """
        Apply the inverse wavelet transform to the Besov weighted signal.

        Args:
            signal (np.ndarray): Input signal to be transformed.

        Returns:
            np.ndarray: Signal transformed using Besov weights.
        """

        # Compute Besov weights
        weight = self.weights()

        # Normalizing Besov weights
        size = np.linalg.norm(weight**-1,ord=2) # Normalization constant
        weight[0:2**(self.level*2)] = size # Normalizing scaling coefficient weights
        weight[2**(self.level*2)::]*= size # Normalizing wavelet coefficient weights

        # Computing the weighted signal
        coeff = weight*signal

        # Performing the inverse wavelet transform on the weighted signal
        return pywt.waverec2(pywt.unravel_coeffs(coeff, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')


    def inv_wavelet_weight_adjoint(self,signal):
        """
        Apply the wavelet transform to the inverse Besov weighted signal.

        Args:
            signal (np.ndarray): Input signal to be transformed.

        Returns:
            np.ndarray: Signal transformed using inverse Besov weights.
        """
        # Computing inverse Besov weights
        inv_weigth = self.inverse_weights()

        # Normlizing the weights
        size = np.linalg.norm(inv_weigth,ord=2) # Normalization constant
        inv_weigth/= size # Normalizing the wavelet coefficient weights
        inv_weigth[0:2**(self.level*2)] = 1/size # Normalizing the scaling coefficient weights

        # Performing the wavelet transform on the signal
        coeff = pywt.wavedec2(signal.reshape((2**self.J,2**self.J)), self.wavelet, mode='periodization', level=self.J-self.level)

        # Returning inverse Besov weighted wavelet coefficients as an array
        return inv_weigth*pywt.ravel_coeffs(coeff)[0]

    


    def prior_to_normal(self, x):
        """
        Transform the Besov prior to a standard normal distribution.

        Args:
            x (np.ndarray): Input data to be transformed.

        Returns:
            tuple: Transformed data and its derivative.
        """
        
        g = np.zeros(len(x))       # Transformed values
        g_diff = np.zeros(len(x))  # First derivatives of the transform

        # Scale parameter for the generalized normal distribution
        scal = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)
        lam = scal ** (-self.p)
        stop = 15  # Threshold for linear approximation

        # Transform values in different regions
        index1 = np.logical_and(x > 0, x < stop)
        index2 = np.logical_and(x <= 0, x > -stop)
        index3 = np.logical_or(x < -stop, x > stop)

        # Region 1: Positive values within threshold
        cdf1 = norm.cdf(-x[index1])
        g[index1] = -gennorm.ppf(cdf1, self.p, loc=0, scale=scal)
        g_diff[index1] = norm.pdf(x[index1]) / (gennorm.pdf(g[index1], self.p, loc=0, scale=scal))

        # Region 2: Negative values within threshold
        cdf2 = norm.cdf(x[index2])
        g[index2] = gennorm.ppf(cdf2, self.p, loc=0, scale=scal)
        g_diff[index2] = norm.pdf(-x[index2]) / (gennorm.pdf(g[index2], self.p, loc=0, scale=scal))

        # Region 3: Outside threshold (linear approximation)
        g[index3] = np.sign(x[index3]) * (np.abs(x[index3]) ** 2 / (2 * lam)) ** (1 / self.p)
        g_diff[index3] = np.abs(x[index3])/(lam*self.p)*(np.abs(x[index3])**2/(2*lam))**(1/self.p-1)
        
        return g, g_diff


    def transform(self,x):
        """
        Apply the full prior transformation B^1g(*).

        Args:
            x (np.ndarray): Input signal.

        Returns:
            np.ndarray: Transformed signal.
        """
        
        # Compute the transformation g
        g = self.prior_to_normal(x)[0]
        
        # Compute B^-1 * g (inverse wavelet transformation with weights)
        return self.inv_wavelet_weigth(g)


    def jac_const(self):
        """
        Compute the Besov matrix B^-1, which is used as the prior Jacobian.

        Returns:
            scipy.sparse.csr_matrix: The sparse representation of the Besov matrix.
        """
        
        N = 2 ** self.J  # Dimension of the input
        I0 = np.zeros((N, N))  # Identity matrix placeholder
        A = np.zeros((N**2, N**2))  # Placeholder for the Besov matrix
        
        # Compute the Besov matrix by applying the linear transform to the identity matrix
        for i in range(N):
            for j in range(N):
                I0[i,j] = 1 # Set a single element to 1
                a_row = self.inv_wavelet_weigth(I0.flatten())  # Apply the transform
                A[:,i*N+j] = a_row.flatten() # Store the result in the Besov matrix
                I0[i,j] = 0 # Reset the element to 0

        # Return the sparse representation of the Besov matrix
        return sparse.csr_matrix(A)
    
    def compute_Besov_matrix(self):
        """
        Compute the Besov matrix B^-1, which is used as the prior Jacobian.

        Returns:
            scipy.sparse.csr_matrix: The sparse representation of the Besov matrix.
        """
        N = 2 ** self.J  # Dimension of the input
        I0 = np.zeros((N, N))  # Identity matrix placeholder
        A = np.zeros((N**2, N**2))  # Placeholder for the Besov matrix
        
        # Compute the Besov matrix by applying the wavelet weights to the identity matrix
        for i in range(N):
            for j in range(N):
                I0[i,j] = 1 # Set a single element to 1
                a_row = self.wavelet_weigth(I0.flatten()) # Apply the wavelet weights
                A[:,i*N+j] = a_row.flatten() # Store the result in the Besov matrix
                I0[i,j] = 0 # Reset the element to 0

        # Return the sparse representation of the Besov matrix
        return sparse.csr_matrix(A)
     
