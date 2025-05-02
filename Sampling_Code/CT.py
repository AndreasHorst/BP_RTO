import numpy as np
from skimage.transform import radon
from scipy import sparse
# The CT problem class
class CT:
    # This class represents a Computed Tomography (CT) problem.
    # It defines the likelihood of an inverse problem using a CT forward map.
    # The problem is characterized by its projection angles (`theta`), 
    # which correspond to the angles at which parallel x-rays are projected through the object.
    def __init__(self, theta,sigma=0, data=0):
        # Initialize the CT problem with the given projection angles, noise level, and data.
        
        # Projection angles for the CT scan
        self.theta = theta 
        
       # Data for the likelihood function (e.g., the observed data)
        self.data = data
        
        # Standard deviation of the noise in the likelihood
        self.lam = sigma


    def forward(self,signal):
        # Compute the Radon transform of the input signal.
        # The Radon transform is used to simulate x-ray projections through the object.
        # `circle=False` ensures that the transform is computed for the entire square image.
        return radon(signal,theta=self.theta, circle=False)

    def set_data(self,signal, noise_level):
        # Generate synthetic data by applying the forward operation on the input signal
        # and adding Gaussian noise based on the specified noise level.

        # Apply the forward operation to the input signal
        Forward_data = self.forward(signal)

        # Compute the noise standard deviation as a fraction of the data norm
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data.flatten(), ord=2)/np.sqrt(len(Forward_data.flatten())))
        
        # Generate Gaussian noise with the computed standard deviation
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        
        # Add the noise to the forward data to create the noisy observation
        self.data = Forward_data + noise



    def jac_const(self,n):
        # Compute the Jacobian matrix (or forward operator) as a sparse matrix.
        # This matrix represents the linear mapping of the input signal to the projections.

        # Initialize an empty matrix for the forward operator
        A = np.zeros((int(np.ceil(np.sqrt(2)*n))*len(self.theta),n**2))

        # Create an identity matrix for the input signal space
        I_0 = np.zeros((n,n))

        # Loop through each element of the input space to compute its contribution
        for i in range(n):
            for j in range(n):   
                I_0[i,j] = 1 # Set a single element to 1
                a_row = self.forward(I_0) # Compute the forward map for this element
                A[:,i*n+j] = a_row.flatten() # Flatten and store in the matrix
                I_0[i,j] = 0  # Reset the element

        # Return the forward operator as a sparse matrix
        return sparse.csr_matrix(A)
