import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft
class blurring:
    def __init__(self,x, sigma_kernel,Factor=1 ,lam=0, data=0):
        self.grid=x
        self.sigma = sigma_kernel
        self.data = data
        self.lam = lam
        self.Factor = Factor

    def kernel(self):
        sigma = self.sigma
        if np.logical_and(0 < sigma, sigma <= 1 / 6):
            # Continuous normalization constant ensuring unit integration of the kernel
            c_sigma = 1 / (2.5 * sigma)
            # Gaussian kernel with standard deviation sigma with domain [0,1)
            c= np.piecewise(self.grid,
                            [self.grid <= 3 * sigma, np.logical_and(3 * sigma <= self.grid, self.grid <= 1 - 3 * sigma),
                            np.logical_and(1 - 3 * sigma < self.grid, self.grid <= 1)],
                            [lambda x: c_sigma * np.exp(-x ** 2 / (2 * sigma ** 2)), lambda x: 0,
                            lambda x: c_sigma * np.exp(-(x - 1) ** 2 / (2 * sigma ** 2))])
            # Discrete normalization
            return  c/ np.linalg.norm(c, ord=1)
        else:
            print("Sigma value is not supported")
            exit()
        

    def forward(self,signal):
        if len(self.kernel()) == len(signal):
            result = ifft(fft(self.kernel())*fft(signal))
        return result.real[0::self.Factor]

    def set_data(self,signal, noise_level):
        Forward_data = self.forward(signal)
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data, ord=2))/np.sqrt(len(Forward_data))
        # Noise
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        # Adding the noise with relative noiselevel (noise_level)
        self.data = Forward_data + noise


    def jac_const(self,N):
        I = np.identity(N)
        return self.forward(I)    

