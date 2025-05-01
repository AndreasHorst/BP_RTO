import numpy as np
# The inpainting problem class
class inpainting:
    # The inpainting problem is defined by its missing_index, that is the indicies it removes from the signal.
    #  The class defines the likelihood of an inverse problem with inpainting forward map
    def __init__(self, missing_index,sigma=0, data=0):
        # Missing index
        self.index = missing_index 
        # Data for the likelihood
        self.data = data
        # standard deviation of the likelihood
        self.lam = sigma
        self.f_eval = 0

        
    # Forward operation
    def forward(self,signal):
        # deletes the indicies
        self.f_eval+=1
        return np.delete(signal,self.index,0)
    
    def forward_adjoint(self,signal):
        n = len(signal)+len(self.index)
        index_remain = np.delete(np.arange(0,n),self.index)
        a = np.zeros(n)
        a[index_remain] = signal
        return a

    def set_data(self,signal, noise_level):
        # set:data generates synthetic data given the ground truth (signal) and adds additive gaussian noise with relative noise_level.
        Forward_data = self.forward(signal)
        self.lam = (noise_level/100)*(np.linalg.norm(Forward_data, ord=2))*1/np.sqrt(len(Forward_data))
        # Noise
        noise = np.random.standard_normal(size=Forward_data.shape)*self.lam
        # Adding the noise with relative noiselevel (noise_level)
        self.data = Forward_data + noise
        # Setting the standard deviation of the likelihood as the standard deviation of the data
        #self.lam = np.std(self.data)



    def jac_const(self,N):
        # Computing the forward operator A as a matrix
        return self.forward(np.identity(N))    
