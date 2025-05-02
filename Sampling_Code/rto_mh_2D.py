import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from scipy.linalg import qr
import scipy.optimize as opt
from scipy import sparse

# The rto_mh_2D class implements the Randomize-Then-Optimize (RTO) sampler
# with a Metropolis-Hastings (MH) acceptance-rejection step for 2D problems.
class rto_mh_2D:
    def __init__(self, x_start, Nrand,Q=0, Q1=0,Q2=0,samp=1000):
         """
        Initialize the RTO-MH sampler.

        Args:
            x_start (np.ndarray): Initial guess for the parameters.
            Nrand (int): Dimension of the realizable normal distribution (n + m).
            Q (scipy.sparse matrix): Matrix for the RTO algorithm (default: identity matrix).
            Q1, Q2 (scipy.sparse matrices): Partitioned components of Q (default: zero).
            samp (int): Number of samples to generate (default: 1000).
        """
        self.x0 = x_start # Starting point for optimization
        self.Nrand = Nrand # Dimension of the realizable normal distribution 
        # Q matrix for the RTO algorithm
        if Q==0:
            self.Q =sparse.identity(Nrand,dtype='float64', format='csr')
        else:
            self.Q=Q
        self.eps = np.zeros(Nrand) # Random Gaussian
        self.nsamp = samp # Number of samples
        self.Q1 = Q1 # First partition of Q
        self.Q2 = Q2 # Second partition of Q


    def residual(self,x, problem):
        """
        Compute the residual for the least squares RTO problem.

        Args:
            x (np.ndarray): Current parameter vector.
            problem: Inverse problem instance.

        Returns:
            np.ndarray: Concatenated residual (forward map residual and parameter residual).
        """
        return np.concatenate((problem.residual(x).flatten(),x),axis=0)


    def cost(self, x, problem):
        """
        Compute the cost function Q.T * (AB^-1g(*) - y - epsilon).

        Args:
            x (np.ndarray): Current parameter vector.
            problem: Inverse problem instance.

        Returns:
            tuple: Transformed residual and raw residual.
        """
        # Computing residual
        r = self.residual(x,problem)

        # Returning cost and residual
        return self.Q@(r-self.eps), r
    

    def jac(self,x,problem):
        """
        Compute the Jacobian of the RTO optimization problem.

        Args:
            x (np.ndarray): Current parameter vector.
            problem: Inverse problem instance.

        Returns:
            scipy.sparse matrix: Jacobian matrix.
        """
        
        # Jacobian of the RTO optimzation problem Q.T*([J_F,I]) using the partition matrices 
        res = self.Q1.multiply(problem.likelihood.lam**-1*problem.prior.prior_to_normal(x)[1])+self.Q2
        
        return res.tocsr()

    def jac_initialize(self,x,problem):
        """
        Initialize the Jacobian matrix for RTO optimization.

        Args:
            x (np.ndarray): Current parameter vector.
            problem: Inverse problem instance.

        Returns:
            scipy.sparse matrix: Initial Jacobian matrix used for computing Q.
        """
        
        # Initial Jacobian of the RTO optimzation problem ([J_F,I])
        return sparse.vstack([problem.jac(x),sparse.identity(len(x),dtype='float64', format='csr')])


    def initialize_Q(self,problem):
        """
        Compute the Q matrix as the Q-factor from the QR decomposition of J_F(MAP).

        Args:
            problem: Inverse problem instance.

        Returns:
            np.ndarray: MAP (Maximum A Posteriori) estimate.
        """
        # Solve for the MAP estimate with initial Q and epsilon set to zero
        Sol = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0,lambda x: self.jac_initialize(x,problem), method='trf',
                               verbose=2, ftol=1e-08, xtol=1e-08, gtol=1e-08,x_scale='jac')
        print(np.linalg.norm(Sol.grad)) # Debug: Gradient norm at solution
        
        # Compute the Q matrix from the QR decomposition of the Jacobian
        Qmatrix = qr(Sol.jac.toarray(), mode='economic')[0]
        
        # Transpose the Q-matrix
        self.Q = sparse.csr_matrix(np.transpose(Qmatrix))

        # Initialize the partitions
        n = len(self.x0)
        self.Q1= self.Q[0:n,0:self.Nrand-n]@problem.jac_const
        self.Q2 = self.Q[0:n,self.Nrand-n:self.Nrand]
        
        # return the MAP estimate
        return Sol.x




    def Compute_log_MH(self,problem):
        """
        Compute the sample and coefficients for the Metropolis-Hastings step.

        Args:
            problem: Inverse problem instance.

        Returns:
            tuple: Logarithm of the MH weight, transformed sample, and residual norm.
        """
        # Generate a random sample from the standard normal distribution
        self.eps= np.random.standard_normal(self.Nrand)
        
        # Solve the stochastic least squares problem to get the proposal sample
        res = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0, lambda x: self.jac(x, problem), method='trf', ftol=1e-04, xtol=1e-04, gtol=1e-04, tr_solver='lsmr', x_scale='jac')
        print(res.cost, np.linalg.norm(res.grad)) # Debug: Cost and gradient norm
        nresid = res.cost # Cost
        x_star = res.x # Proposal sample

        
        self.eps = np.zeros(self.Nrand) # Reset to zero
        
        # Compute the cost, residual, and Jacobian at the proposal sample
        Qtr, r = self.cost(x_star, problem)
        QtJ = self.jac(x_star, problem)
        R = np.linalg.qr(QtJ.toarray(), mode='r')

        # Compute MH weight using QR decomposition
        log_c_temp = np.sum(np.log(np.abs(np.diag(R)))) + 0.5 * (np.linalg.norm(r, ord=2) ** 2 - np.linalg.norm(Qtr,ord=2) ** 2)
        
        # Transform the sample back to the original domain
        x = problem.prior.transform(x_star).flatten()    
        
        return log_c_temp, x, nresid




    def sample(self,problem):
        """
        Generate RTO samples in parallel and perform Metropolis-Hastings acceptance-rejection.

        Args:
            problem: Inverse problem instance.

        Returns:
            tuple: Chain of samples, acceptance rate, accepted indices, log weights.
        """
        # Initialize parameters
        Nrand = self.Nrand
        num_cores = multiprocessing.cpu_count()
        N = len(self.x0)

        # Initialize the sample chain
        xchain = np.zeros((N,self.nsamp+1))
        xchain[:,0] = self.x0
        
        # Evaluate the cost, residual, and Jacobian for the initial guess
        self.eps = np.zeros(Nrand)
        Qtr, r = self.cost(self.x0, problem)
        QtJ = self.jac(self.x0, problem)
        
        # Compute the log determinant using QR decomposition
        R = np.linalg.qr(QtJ.toarray(), mode='r')
        log_c_chain = np.zeros(self.nsamp+1)
        log_c_chain[0] =np.sum(np.log(np.abs(np.diag(R)))) +0.5*(np.linalg.norm(r,ord=2)**2- np.linalg.norm(Qtr,ord=2)**2)

        n_accept = 0 # Acceptance counter

        # Parallel computation of MH proposals
        L = Parallel(n_jobs=num_cores)(delayed(self.Compute_log_MH)(problem)
                                                    for i in range(self.nsamp))

        # Process results
        log_c_temp = np.zeros(self.nsamp)
        nresid = np.zeros(self.nsamp)
        x_star = np.zeros((N,self.nsamp))
        count = 0
        for i in L:
            log_c_temp[count] = i[0]
            x_star[:,count] = i[1]
            nresid[count] = i[2]
            count+=1
            
        # Sequential MH acceptance-rejection
        index_accept = []    
        for i in range(self.nsamp):
            # Determining by MH ratio if the sample should be accepted or rejected
            if np.logical_and(log_c_chain[i]-log_c_temp[i] > np.log(np.random.uniform(0,1,1)),nresid[i]< 1e-08):
                index_accept.append(i)
                n_accept += 1
                xchain[:,i+1] = x_star[:,i]
                log_c_chain[i+1]=log_c_temp[i]
            else:
                xchain[:,i+1] = xchain[:, i]
                log_c_chain[i+1] = log_c_chain[i]
                
        # Remove initial guess from chain and compute acceptance ratio
        xchain = np.delete(xchain, 0, axis=1)
        acc_rate = n_accept/self.nsamp
        return xchain, acc_rate, index_accept, log_c_chain        
