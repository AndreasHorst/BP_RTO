import numpy as np
import pywt
from scipy.stats import gennorm, norm
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.special import gamma
# Class that defines the Besov prior with all its functionalities
class besov_prior2D:
    def __init__(self,J,delt,level, slices, shape ,s=1, p=1, wavelet="db1"):
        # s paramter
        self.s = s
        # p parameter
        self.p = p
        # wavelet basis
        self.wavelet = wavelet
        # Maximum amount of wavelet levels, that is dimension n=2^(2*J).
        self.J = J
        # Prior regularization paramter
        self.delt = delt
        # Level determines the amount of levels to not use in the Discrete wavelet transform
        self.level = level
        # list that specifies the structure of of the 2D wavelet coefficients such that we can transform back and forth between 2D and 1D coefficients structures.
        self.coeff_slices = slices
        self.coeff_shape = shape



    def sample(self):
        # Compute random draws
        # Standard deviation
        scale = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)
        xi = gennorm.rvs(self.p, scale=scale, size=2**(self.J*2))
        # Compute coefficient weights
        weights = self.inverse_weights()
        # Compute expansion coefficients
        coefficients = (weights/np.linalg.norm(weights,ord=2))*xi
        # Compute the sample from a Besov prior
        return pywt.waverec2(pywt.unravel_coeffs(coefficients, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')

    
    

    def inverse_weights(self):
        # Dimension
        n = 2**(self.J*2)
        # Allocation
        inv_weights = np.zeros(n)
        # Computing scaling function weigth
        inv_weights[0:2**((self.level)*2)] = 2**(-self.level*(self.s+1.0-2.0/self.p))
        # Computing the weights for each wavelet level.
        for i in range(self.level,self.J):
            inv_weights[2**(i*2):2**((i+1)*2)]=2**(-i*(self.s+1.0-2.0/self.p))
        return inv_weights



    def weights(self):
        n = 2**(self.J*2)
        weights = np.zeros(n)
        weights[0] = 1
        for i in range(self.J):
            weights[2**(i*2):2**((i+1)*2)]=2**(i*(self.s+1.0-2.0/self.p))
        return weights          
    
    def inv_wavelet_weigth(self, signal):
        # Computing the inverse Besov weights
        inv_weigth = self.inverse_weights()
        size = np.linalg.norm(inv_weigth,ord=2)
        coeff = (inv_weigth/size)*signal
        coeff[0:2**(self.level*2)]=1/size*signal[0:2**(self.level*2)]
        return pywt.waverec2(pywt.unravel_coeffs(coeff, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')
    

    
    def wavelet_weigth(self, signal):
         weights = self.weights()
         wavelet_coefficients = pywt.wavedec2(signal.reshape((2**self.J,2**self.J)), self.wavelet, mode='periodization', level=self.J-self.level)
         size = np.linalg.norm(weights**-1,ord=2)
         weights[0:2**(self.level*2)] = size
         weights[2**(self.level*2)::]*= size
         return weights*pywt.ravel_coeffs(wavelet_coefficients)[0]
    
    def wavelet_weight_adjoint(self,signal):
        weight = self.weights()
        size = np.linalg.norm(weight**-1,ord=2)
        weight[0:2**(self.level*2)] = size
        weight[2**(self.level*2)::]*= size
        coeff = weight*signal
        return pywt.waverec2(pywt.unravel_coeffs(coeff, self.coeff_slices, self.coeff_shape, output_format='wavedec2'), self.wavelet, mode='periodization')


    def inv_wavelet_weight_adjoint(self,signal):
        inv_weigth = self.inverse_weights()
        size = np.linalg.norm(inv_weigth,ord=2)
        inv_weigth/= size 
        inv_weigth[0:2**(self.level*2)] = 1/size
        coeff = pywt.wavedec2(signal.reshape((2**self.J,2**self.J)), self.wavelet, mode='periodization', level=self.J-self.level)
        return inv_weigth*pywt.ravel_coeffs(coeff)[0]

    


    def prior_to_normal(self, x):
        # prior_to_normal computes the map g and its derivative that transforms a p-norm prior to a standard Gaussian.
        # Allocation
        g = np.zeros(len(x))
        g_diff = np.zeros(len(x))
        # Splitting the indicies in two categories. Index 1 is where we have to approximate the map. Index 2 is where we can compute the map directly.
        index3 = np.logical_or(x<-15, x>15)
        scal = np.sqrt(gamma(1/self.p)/gamma(3/self.p))*self.delt**(-1/self.p)
        # Regularization paramter delt/p
        lam = scal**(-self.p)
        # Computing the cdf of the standard normal distribution
        index1 = np.logical_and(x > 0,x < 15)
        index2 = np.logical_and(x <= 0, x > -15)
        cdf1 = norm.cdf(-x[index1])
        cdf2 = norm.cdf(x[index2])
        g[index1] = -gennorm.ppf(cdf1, self.p, loc=0, scale=scal)
        g[index2] = gennorm.ppf(cdf2, self.p, loc=0, scale=scal)
        g_diff[index1] = norm.pdf(x[index1]) / (gennorm.pdf(g[index1], self.p, loc=0, scale=scal))
        g_diff[index2] = norm.pdf(-x[index2]) / (gennorm.pdf(g[index2], self.p, loc=0, scale=scal))
        # Computing the approximation for indicies where we do not have enough numerical precision
        g[index3] = np.sign(x[index3]) * (np.abs(x[index3]) ** 2 / (2 * lam)) ** (1 / self.p)
        g_diff[index3] = np.abs(x[index3])/(lam*self.p)*(np.abs(x[index3])**2/(2*lam))**(1/self.p-1)
        # Returning the computed results
        return g, g_diff


    def transform(self,x):
        # Transform computes the full prior transformation B^1g(*)
        # Computing g
        g = self.prior_to_normal(x)[0]
        # Computing B^-1*g
        return self.inv_wavelet_weigth(g)


    def jac_const(self):
        # Computing the Besov matrix B^-1 which is used as the prior jacobian.
        N = 2**self.J
        I0 = np.zeros((N,N))
        A = np.zeros((N**2,N**2))
        # Evaluating the linear transform on the identity matrix provides the matrix B^-1.
        for i in range(N):
            for j in range(N):
                I0[i,j] = 1
                a_row = self.inv_wavelet_weigth(I0.flatten()) 
                A[:,i*N+j] = a_row.flatten()
                I0[i,j] = 0 
        return sparse.csr_matrix(A)
    
    def compute_Besov_matrix(self):
            # Computing the Besov matrix B^-1 which is used as the prior jacobian.
        N = 2**self.J
        I0 = np.zeros((N,N))
        A = np.zeros((N**2,N**2))
        # Evaluating the linear transform on the identity matrix provides the matrix B^-1.
        for i in range(N):
            for j in range(N):
                I0[i,j] = 1
                a_row = self.wavelet_weigth(I0.flatten()) 
                A[:,i*N+j] = a_row.flatten()
                I0[i,j] = 0 
        return sparse.csr_matrix(A)
     
    
    def compute_inv_Besov_matrix(self):
            # Computing the Besov matrix B^-1 which is used as the prior jacobian.
        N = 2**self.J
        I0 = np.zeros((N,N))
        A = np.zeros((N**2,N**2))
        # Evaluating the linear transform on the identity matrix provides the matrix B^-1.
        for i in range(N):
            for j in range(N):
                I0[i,j] = 1
                a_row = self.inv_wavelet_weigth(I0.flatten()) 
                A[:,i*N+j] = a_row.flatten()
                I0[i,j] = 0 
        return sparse.csr_matrix(A)
