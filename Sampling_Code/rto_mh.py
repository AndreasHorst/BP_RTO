import numpy as np
from scipy.linalg import qr
import scipy.optimize as opt
# The rto_mh class defines the sampler Randomize-Then-Optimize (RTO) 
# with a Metropolis-Hastings (MH) acceptance-rejection step.
class rto_mh:
    def __init__(self, x_start, Nrand,Q=0,samp=1000):
        # Initialize the RTO-MH sampler with the given parameters.
        
        # Initial guess for the parameter vecto
        self.x0 = x_start
        
        # Dimensionality of the realizable normal distribution (typically n + m)
        self.Nrand = Nrand
        
        # Q matrix for the RTO algorithm (default is identity matrix if not provided)
        if Q==0:
            self.Q =np.identity(Nrand)
        else:
            self.Q=Q
            
        # Random draw of the standard normal distribution
        self.eps = np.zeros(Nrand)
        
        # Number of samples to generate
        self.nsamp = samp



    def residual(self,x, problem):
        # Compute the residual for the least squares RTO problem by concatenating:
        # - The residual of the problem
        # - The standard Gaussian prior
        return np.concatenate((problem.residual(x),x),axis=0)


    def cost(self, x, problem):
        # Compute the cost function: Q.T @ (residual - epsilon)
        res = self.residual(x,problem)
        return self.Q@(res-self.eps), res

    def jac(self,x,problem):
        # Compute the Jacobian of the RTO problem
        return self.Q@np.concatenate([problem.jac(x),np.identity(len(x))],axis=0)

    def initialize_Q(self,problem):
       # Initialize the Q matrix using the QR decomposition of the Jacobian at the MAP estimate.
        
        # Solve the least squares problem to find the MAP estimate (Q = identity, epsilon = 0)
        Sol = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0,jac= lambda x: self.jac(x,problem) ,method='lm',
                                verbose=2,ftol=1e-8, xtol=1e-8, gtol=1e-8)
        
        print(np.linalg.norm(Sol.grad)) # Print the gradient norm          
        
       # Perform QR decomposition of the Jacobian
        Qmatrix, R = qr(Sol.jac, mode='economic')
        # Transpose the Q-matrix
        self.Q = np.transpose(Qmatrix)
        
        # Return the MAP estimate
        return Sol.x


    def Compute_log_MH(self,problem):
        # Compute the proposal sample and associated coefficients for the 
        # Metropolis-Hastings acceptance-rejection step.
        
        # Sample from the standard normal distribution
        self.eps= np.random.standard_normal(self.Nrand)
        
        # Solve the stochastic least squares problem to get the proposal sample
        res = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0, lambda x: self.jac(x, problem), method='lm', ftol=1e-06, xtol=1e-06, gtol=1e-06)
        # Extract the residual and the proposal sample
        nresid = res.cost
        x_star = res.x
        # Reset epsilon
        self.eps = np.zeros(self.Nrand)
        
        # Compute the cost, residual, and Jacobian at the proposal sample
        Qtr, r = self.cost(x_star, problem)
        QtJ = self.jac(x_star, problem)

        # Compute the R factor of the QR decomposition
        R =np.linalg.qr(QtJ, mode='r')
        
        # Compute the log coefficient for the MH ratio
        log_c_temp = np.sum(np.log(np.abs(np.diag(R)))) + 0.5 * (np.linalg.norm(r, ord=2) ** 2 - np.linalg.norm(Qtr,ord=2) ** 2)   

        return log_c_temp, problem.prior.transform(x_star), nresid




    def sample(self,problem):
        # Generate samples using the RTO-MH algorithm.
        # The Metropolis-Hastings acceptance-rejection step is performed in sequence.

        # Initialize parameters
        Nrand = self.Nrand
        N = len(self.x0)
        xchain = np.zeros((N,self.nsamp+1)) # Chain of samples
        xchain[:,0] = self.x0 # Set the initial guess
        
        # Evaluate the cost, residual, and Jacobian at the initial guess
        self.eps = np.zeros(Nrand)
        Qtr, r = self.cost(self.x0, problem)
        QtJ = self.jac(self.x0, problem)

        # Initialize log coefficients and acceptance parameters
        log_c_chain = np.zeros(self.nsamp+1)
        log_c_chain[0] = np.sum(np.log(np.abs(np.diag(np.linalg.qr(QtJ, mode='r')))))+0.5*(np.linalg.norm(r,ord=2)**2
        - np.linalg.norm(Qtr,ord=2)**2)
        n_accept = 0 # Counter for accepted samples
        log_c_temp = np.zeros(self.nsamp)
        nresid = np.zeros(self.nsamp)
        x_star = np.zeros((N,self.nsamp)) # Array for proposal samples

        # Compute proposal samples and associated log coefficients
        for i in range(self.nsamp):
            log_c_temp[i], x_star[:,i], nresid[i] = self.Compute_log_MH(problem)

        index_accept = [] # Indices of accepted samples

        # Perform the Metropolis-Hastings acceptance-rejection step
        for i in range(self.nsamp):
            # Compute the MH ratio and check acceptance condition
            if np.logical_and(log_c_chain[i]-log_c_temp[i] > np.log(np.random.uniform(0,1,1)),nresid[i]< 1e-08):
                index_accept.append(i)
                n_accept += 1
                xchain[:,i+1] = x_star[:,i] # Accept the sample
                log_c_chain[i+1]=log_c_temp[i]
            else:
                xchain[:,i+1] = xchain[:, i] # Reject the sample
                log_c_chain[i+1] = log_c_chain[i]
                
        # Remove the initial guess from the chain and compute the acceptance ratio
        xchain = np.delete(xchain, 0, axis=1)
        acc_rate = n_accept/self.nsamp
        return xchain, acc_rate, index_accept, log_c_chain        
