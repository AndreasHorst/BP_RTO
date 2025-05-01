import numpy as np

class inverse_problem:
    # The inverse problem class sets up the inverse problem in a Bayesian form with prior, likelihood and jacobian needed for the RTO sampler
    def __init__(self,likelihood, prior, jac_const):
        # likelihood class
        self.likelihood = likelihood
        # Prior class
        self.prior = prior
        # Jacobian method
        self.jac_const = jac_const

    def residual(self,x):
        # Residual computes the missmatch between the forward map of the signal and the data.
        return (1/self.likelihood.lam)*(self.likelihood.forward(self.prior.transform(x))-self.likelihood.data)

    def jac(self,x):
        # Compute the jacobian including the transform jacobian.
        gdiff = self.prior.prior_to_normal(x)[1]
        return (1/self.likelihood.lam)*gdiff*self.jac_const

