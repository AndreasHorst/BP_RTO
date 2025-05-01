import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from scipy.linalg import qr
import scipy.optimize as opt
from scipy import sparse

class rto_mh_2D:
    # The rto_mh class defines the sampler Randomize-Then-Optimize with a Metropolis-Hastings acceptance-rejection step.
    def __init__(self, x_start, Nrand,Q=0, Q1=0,Q2=0,samp=1000):
        # Start guess
        self.x0 = x_start
        # Dimension of the realizable normal distribution typically n+m (dimension of unknown + dimension of data)
        self.Nrand = Nrand
        # Q matrix for the RTO algorithm
        if Q==0:
            self.Q =sparse.identity(Nrand,dtype='float64', format='csr')
        else:
            self.Q=Q
        # Draw of the realizable gaussian distribution    
        self.eps = np.zeros(Nrand)
        # Amount of samples
        self.nsamp = samp
        self.Q1 = Q1
        self.Q2 = Q2




    def residual(self,x, problem):
        # Residual function for the least squares RTO problem. 
        return np.concatenate((problem.residual(x).flatten(),x),axis=0)


    def cost(self, x, problem):
        # Cost function Q.T*(AB^-1g(*)-y-epsilon). 
        r = self.residual(x,problem)
        return self.Q@(r-self.eps), r
    

    def jac(self,x,problem):
        # Jacobian of the RTO optimzation problem Q.T*([J_F,I]). 
        res = self.Q1.multiply(problem.likelihood.lam**-1*problem.prior.prior_to_normal(x)[1])+self.Q2
        return res.tocsr()

    def jac_initialize(self,x,problem):
        # Jacobian of the RTO optimzation problem Q.T*([J_F,I])
        return sparse.vstack([problem.jac(x),sparse.identity(len(x),dtype='float64', format='csr')])


    def initialize_Q(self,problem):
        # Procedure to find Q as the Q matrix of the QR decomposition of J_{F}(MAP).
        # Compute the map with Q = identity and epsilon= 0
        # Compute Map argmin||IF(x)-0||_2^2
        Sol = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0,lambda x: self.jac_initialize(x,problem), method='trf',
                               verbose=2, ftol=1e-08, xtol=1e-08, gtol=1e-08,x_scale='jac')
        print(np.linalg.norm(Sol.grad))
        # Set the Q matrix
        Qmatrix = qr(Sol.jac.toarray(), mode='economic')[0]
        # Transpose the Q-matrix
        self.Q = sparse.csr_matrix(np.transpose(Qmatrix))
        n = len(self.x0)
        self.Q1= self.Q[0:n,0:self.Nrand-n]@problem.jac_const
        self.Q2 = self.Q[0:n,self.Nrand-n:self.Nrand]
        # return the map estimate
        return Sol.x




    def Compute_log_MH(self,problem):
        # This function computes the samples and the coefficients used for the Metropolis-Hastings acceptance-rejection step
       # Computing random sample from standard normal distribution $\nu$
        self.eps= np.random.standard_normal(self.Nrand)
        # Solving the stochastic least squares problem to get the proposal sample 
        res = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0, lambda x: self.jac(x, problem), method='trf', ftol=1e-04, xtol=1e-04, gtol=1e-04, tr_solver='lsmr', x_scale='jac')
        print(res.cost, np.linalg.norm(res.grad))
        nresid = res.cost
        x_star = res.x
        self.eps = np.zeros(self.Nrand)
        # Computing the cost, residual, and Jacobian at the proposal sample
        # Compuuting each factor in the Metropolis-Hastings weight at the proposal sample 
        Qtr, r = self.cost(x_star, problem)
        QtJ = self.jac(x_star, problem)
        R = np.linalg.qr(QtJ.toarray(), mode='r')
        log_c_temp = np.sum(np.log(np.abs(np.diag(R)))) + 0.5 * (np.linalg.norm(r, ord=2) ** 2 - np.linalg.norm(Qtr,ord=2) ** 2)
        #Transforming the sample back to original domain
        x = problem.prior.transform(x_star).flatten()                                                                                     
        return log_c_temp, x, nresid




    def sample(self,problem):
        # This function computes the RTO samples in parallel and makes the Metropolis-Hastings acceptance rejection in sequence
    # initializing parameters
        Nrand = self.Nrand
        num_cores = multiprocessing.cpu_count()
        N = len(self.x0)
        # initializing sample chain
        xchain = np.zeros((N,self.nsamp+1))
        xchain[:,0] = self.x0
        # evaluating the cost, residual- and Jacobian function on the initial guess
        self.eps = np.zeros(Nrand)
        Qtr, r = self.cost(self.x0, problem)
        QtJ = self.jac(self.x0, problem)
        #lu = sparse.linalg.splu(QtJ.tocsc())
        #Computing log determinant using QR factorization
        R = np.linalg.qr(QtJ.toarray(), mode='r')
        log_c_chain = np.zeros(self.nsamp+1)
        #Computing log weight of equation (37)
        log_c_chain[0] =np.sum(np.log(np.abs(np.diag(R)))) +0.5*(np.linalg.norm(r,ord=2)**2- np.linalg.norm(Qtr,ord=2)**2)
        n_accept = 0
        L = Parallel(n_jobs=num_cores)(delayed(self.Compute_log_MH)(problem)
                                                    for i in range(self.nsamp))
        log_c_temp = np.zeros(self.nsamp)
        nresid = np.zeros(self.nsamp)
        x_star = np.zeros((N,self.nsamp))
        count = 0
        for i in L:
            log_c_temp[count] = i[0]
            x_star[:,count] = i[1]
            nresid[count] = i[2]
            count+=1
        #The RTO-MH sample for loop
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
        # Deleting initial guess from chain and computing the acceptance ratio.
        xchain = np.delete(xchain, 0, axis=1)
        acc_rate = n_accept/self.nsamp
        return xchain, acc_rate, index_accept, log_c_chain        