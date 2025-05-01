import numpy as np
from scipy.linalg import qr
import scipy.optimize as opt

class rto_mh:
    # The rto_mh class defines the sampler Randomize-Then-Optimize with a Metropolis-Hastings acceptance-rejection step.
    def __init__(self, x_start, Nrand,Q=0,samp=1000):
        # Start guess
        self.x0 = x_start
        # Dimension of the realizable normal distribution typically n+m (dimension of unknown + dimension of data)
        self.Nrand = Nrand
        # Q matrix for the RTO algorithm
        if Q==0:
            self.Q =np.identity(Nrand)
        else:
            self.Q=Q
        # Draw of the realizable gaussian distribution    
        self.eps = np.zeros(Nrand)
        # Amount of samples
        self.nsamp = samp



    def residual(self,x, problem):
        # Residual function for the least squares RTO problem. Adding the residual and the standard Gaussian prior into one vector.
        return np.concatenate((problem.residual(x),x),axis=0)


    def cost(self, x, problem):
        # Cost function Q.T*(AB^-1g(*)-y-epsilon)
        res = self.residual(x,problem)
        return self.Q@(res-self.eps), res

    def jac(self,x,problem):
        return self.Q@np.concatenate([problem.jac(x),np.identity(len(x))],axis=0)

    def initialize_Q(self,problem):
        # Procedure to find Q as the Q matrix of the QR decomposition of J_{F}(MAP).
        # Compute the map with Q = identity and epsilon= 0
        Sol = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0,jac= lambda x: self.jac(x,problem) ,method='lm',
                                verbose=2,ftol=1e-8, xtol=1e-8, gtol=1e-8)
        print(np.linalg.norm(Sol.grad))                 
        # Set the Q matrix
        Qmatrix, R = qr(Sol.jac, mode='economic')
        # Transpose the Q-matrix
        self.Q = np.transpose(Qmatrix)
        # return the map estimate
        return Sol.x


    def Compute_log_MH(self,problem):
        # This function computes the samples and the coefficients used for the Metropolis-Hastings acceptance-rejection step
       # Computing random sample from standard normal distribution
        self.eps= np.random.standard_normal(self.Nrand)
        # Solving the stochastic least squares problem to get the proposal sample
        res = opt.least_squares(lambda x: self.cost(x, problem)[0], self.x0, lambda x: self.jac(x, problem), method='lm', ftol=1e-06, xtol=1e-06, gtol=1e-06)
        nresid = res.cost
        x_star = res.x
        self.eps = np.zeros(self.Nrand)
        # Computing the cost, residual, and Jacobian at the proposal sample
        Qtr, r = self.cost(x_star, problem)
        QtJ = self.jac(x_star, problem)
        R =np.linalg.qr(QtJ, mode='r')
        log_c_temp = np.sum(np.log(np.abs(np.diag(R)))) + 0.5 * (np.linalg.norm(r, ord=2) ** 2 - np.linalg.norm(Qtr,ord=2) ** 2)                                                                                                     
        return log_c_temp, problem.prior.transform(x_star), nresid




    def sample(self,problem):
        # This function computes the RTO samples in parallel and makes the Metropolis-Hastings acceptance rejection in sequence
    # initializing parameters
        Nrand = self.Nrand
        N = len(self.x0)
        xchain = np.zeros((N,self.nsamp+1))
        xchain[:,0] = self.x0
        # evaluating the cost, residual- and Jacobian function on the initial guess
        self.eps = np.zeros(Nrand)
        Qtr, r = self.cost(self.x0, problem)
        QtJ = self.jac(self.x0, problem)
        log_c_chain = np.zeros(self.nsamp+1)
        log_c_chain[0] = np.sum(np.log(np.abs(np.diag(np.linalg.qr(QtJ, mode='r')))))+0.5*(np.linalg.norm(r,ord=2)**2
        - np.linalg.norm(Qtr,ord=2)**2)
        n_accept = 0
        log_c_temp = np.zeros(self.nsamp)
        nresid = np.zeros(self.nsamp)
        x_star = np.zeros((N,self.nsamp))
        for i in range(self.nsamp):
            log_c_temp[i], x_star[:,i], nresid[i] = self.Compute_log_MH(problem)

        index_accept = []

        #The RTO-MH sample for loop
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