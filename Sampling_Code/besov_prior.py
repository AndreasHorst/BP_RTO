import numpy as np
import pywt # Python Wavelet Transform library
from scipy.stats import gennorm, norm
from scipy.special import gamma

# Class that defines the Besov prior with all its functionalities
class besov_prior:
    def __init__(self,J,delt,level, s=1, p=1, wavelet="db1",):
        # Initialize the Besov prior with given parameters.
        
        # Smoothness parameter of the Besov space
        self.s = s
        
        # Regularity parameter of the Besov space
        self.p = p
        
        # Wavelet basis 
        self.wavelet = wavelet
        
        # Maximum number of wavelet levels (dimension is 2^J)
        self.J = J
        
        # Regularization parameter for the prior
        self.delt = delt
        
        # Levels to exclude from the discrete wavelet transform
        self.level = level
    
    def sample(self):
        # Generate a random sample from the Besov prior.

        # Compute the scale of the generalized normal distribution
        scale = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)

        # Sample coefficients from the generalized normal distribution
        xi = gennorm.rvs(self.p, scale=scale, size=2**(self.J))
        
        # Compute the wavelet coefficient weights
        weights = self.inverse_weights()
        
        # Scale the coefficients
        coefficients = weights/np.linalg.norm(weights,ord=2)*xi
        
        # Assemble the wavelet coefficients
        List = []
        List.append(coefficients[0:2**(self.level)]) # Scaling coefficients
        for j in range(self.level,self.J):
         List.append(coefficients[2 ** j:2 ** (j + 1)]) # Wavelet coefficients at each level

        # Perform the inverse wavelet transform to generate the sample
        return pywt.waverec(List, self.wavelet, mode='periodization')
    
    

    def inverse_weights(self):
        # Compute the inverse wavelet weights for the Besov prior.

        
        # Total dimension
        n = 2**self.J
        
        # Allocate space for weights
        inv_weights = np.zeros(n)
        
        # Compute weights for the scaling function
        inv_weights[0:2**self.level] = 2**(-self.level*(self.s+0.5-1.0/self.p))
        
        # Compute weights for each wavelet level
        for i in range(self.level,self.J):
            inv_weights[2**i:2**(i+1)]=2**(-i*(self.s+0.5-1/self.p))
            
        return inv_weights 
    

    
    def inv_wavelet_weigth(self, signal):
        # Apply the inverse wavelet transform of the inverse Besov scaled signal.

        # Compute the inverse Besov weights
        inv_weigth = self.inverse_weights()
        
        # Scale and normalize the signal
        coeff = inv_weigth/np.linalg.norm(inv_weigth,ord=2)*signal
        
        # Assemble the coefficients for the inverse wavelet transform
        List = []
        List.append(signal[0:2**(self.level)]/np.linalg.norm(inv_weigth,ord=2))  # Scaling coefficients
        for j in range(self.level,self.J):
         List.append(coeff[2 ** j:2 ** (j + 1)]) # Wavelet coefficients at each level
            
        # Perform the inverse wavelet transform
        return pywt.waverec(List, self.wavelet, mode='periodization')
    

    
    def inv_wavelet_weight_adjoint(self,signal):
        # Apply the wavelet transform of the inverse Besov scaled signal.

        # Compute the inverse Besov weights
        inv_weigth = self.inverse_weights()

        # Scale and normalize the signal
        size = np.linalg.norm(inv_weigth,ord=2) #scale
        inv_weigth/= size  #Normalization
        inv_weigth[0:2**(self.level)] = 1/size #Normalization

        # Compute wavelet coefficients
        coeff = pywt.wavedec(signal, self.wavelet, mode='periodization', level=self.J-self.level)

        # Return Besov scaled wavelet coefficients as an array
        return inv_weigth*pywt.ravel_coeffs(coeff)[0]


    
    def prior_to_normal(self, x):
        # Transform the Besov prior to a standard normal distribution.

        
        g = np.zeros(len(x)) # Transformed values
        g_diff = np.zeros(len(x)) # First derivatives of the transform

        # Scale parameter for the generalized normal distribution
        scal = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)
        lam = scal**(-self.p)
        stop = 15 # Threshold for exact computations

        # Transform values in different regions
        index1 = np.logical_and(x > 0,x < stop)
        index2 = np.logical_and(x <= 0, x > -stop)
        index3 = np.logical_or(x<-stop, x>stop)

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

        # Return transformed values and first derivatives
        return g, g_diff


    def transform(self,x):
        # Apply the full Besov prior transformation B^-1 * g(*)

        # Transform to standard normal
        g = self.prior_to_normal(x)[0]
        
        # Apply the Besov scaled inverse wavelet transform
        return self.inv_wavelet_weigth(g)


    def jac_const(self,N):
        # Compute the inverse Besov matrix B^-1 

        # Initialize identity and Besov matrix
        I = np.identity(N) # Identity
        A = np.zeros((N,N)) # Besov matrix
        
        # Evaluating the linear transform on the identity matrix 
        for i in range(N):
            A[:,i] = self.inv_wavelet_weigth(I[:,i])  

        Return the inverse Besov matrix
        return A    


    def weights(self):
        # Compute the wavelet weights for the Besov prior.
        
        # Total dimension
        n = 2**self.J
        
        # Allocate space for weights
        weights = np.zeros(n)
        
        # Compute weights for the scaling function
        weights[0] = 1
        
        # Compute weights for each wavelet level
        for i in range(self.J):
            weights[2**i:2**(i+1)]=2**(i*(self.s+0.5-1/self.p))
            
        return weights 


    def wavelet_weigth(self, signal):
        # Apply the Besov weights to the wavelet transformed signal.

        # Perform the wavelet transform on the input signal.
         wavelet_coefficients = pywt.wavedec(signal, self.wavelet, mode='periodization', level=self.J)

        # Compute the Besov weights
         weights = self.weights()

        # Scale and normalize the weights.
         size = np.linalg.norm(self.inverse_weights(),ord=2) # Normalization constant
         weights[0:2**(self.level)] = size
         weights[2**(self.level)::]*=size

        # Return the Besov weighted wavelet coeffcients as an array.
         return weights*pywt.coeffs_to_array(wavelet_coefficients)[0]
    

    def wavelet_weight_adjoint(self,signal):
        # Apply the inverse wavelet transform of the Besov scaled signal.

        # Compute the Besov weights
         weights = self.weights()

        # Scale and normalize the signal
         size = np.linalg.norm(self.inverse_weights(),ord=2) #scale
         weights[0:2**(self.level)] = size # Normalization
         weights[2**(self.level)::]*=size #  Normalization

        # Compute Besov scaled signal
         coeff =weights*signal

        # Assemble the coefficients for the inverse wavelet transform
         List = []
         List.append(signal[0:2**(self.level)]*size) # Scaling coefficients
         for j in range(self.level,self.J):
          List.append(coeff[2 ** j:2 ** (j + 1)]) # Wavelet coefficients at each level
             
        # Performing the inverse wavelet transform on the Besov scaled signal
         return pywt.waverec(List, self.wavelet, mode='periodization')


    def compute_besov_matrix(self):
        # Compute the Besov matrix B 

        # Total dimension
        n = 2**self.J

        # Initilization identity and Besov matrix
        I = np.identity(n) # Identity
        B = np.zeros((n,n)) # Besov matrix

        # Evaluating the linear transform on the identity matrix 
        for i in range(n):
            B[:,i]= self.wavelet_weigth(I[:,i])
            
        #Returning the Besov matrix     
        return B
    
 




        

