import numpy as np
from skimage.transform import radon
from scipy import sparse
# The CT problem class
class CT:
    # The CT problem is defined by its projection angles theta, that is the angles at which parallel x-rays are projected through the object.
    #  The class defines the likelihood of an inverse problem with CT forward map
    def __init__(self, theta,sigma=0, data=0):
        # Projection angles
        self.theta = theta 
        # Data for the likelihood
        self.data = data
        # standard deviation of the likelihood
        self.lam = sigma


    def forward(self,signal):
        return radon(signal,theta=self.theta, circle=False)

    def set_data(self,signal, noise_level):
        # set:data generates synthetic data given the ground truth (signal) and adds additive gaussian noise with relative noise_level.
        #Forward_data = self.forward_2(signal)
        Forward_data = self.forward(signal)
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data.flatten(), ord=2)/np.sqrt(len(Forward_data.flatten())))
        # Noise
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        # Adding the noise with relative noiselevel (noise_level)
        self.data = Forward_data + noise



    def jac_const(self,n):
        # Computing the forward operator A as a matrix
        #A = np.zeros((n*len(self.theta),n**2))
        A = np.zeros((int(np.ceil(np.sqrt(2)*n))*len(self.theta),n**2))
        I_0 = np.zeros((n,n))
        for i in range(n):
            for j in range(n):   
                I_0[i,j] = 1
                a_row = self.forward(I_0)
                A[:,i*n+j] = a_row.flatten()
                I_0[i,j] = 0      
        return sparse.csr_matrix(A)
