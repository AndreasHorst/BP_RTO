import numpy as np

# The inverse_problem class sets up the inverse problem in a Bayesian framework.
# It includes the prior, likelihood, and Jacobian needed for the RTO sampler.
class inverse_problem:
    def __init__(self,likelihood, prior, jac_const):
                """
        Initialize the inverse problem.

        Args:
            likelihood: An object representing the likelihood, containing the forward map, data, and noise level.
            prior: An object representing the prior, including transformations and weights.
            jac_const: The Jacobian matrix (or function) representing the linear operator.
        """
        self.likelihood = likelihood   # likelihood class
        self.prior = prior # Prior class
        self.jac_const = jac_const # Precomputed or provided Jacobian matrix

    def residual(self,x):
        """
        Compute the residual, which represents the mismatch between
        the forward map of the signal and the observed data.

        Args:
            x (np.ndarray): The current parameter vector.

        Returns:
            np.ndarray: The residual vector.
        """
        # Transform the parameters using the prior and compute the forward map
        forward_result = self.likelihood.forward(self.prior.transform(x))

        # Compute the residual by normalizing the mismatch with the noise level
        return (1 / self.likelihood.lam) * (forward_result - self.likelihood.data)

    def jac(self,x):
        """
        Compute the Jacobian matrix, including the contribution from the prior transformation.

        Args:
            x (np.ndarray): The current parameter vector.

        Returns:
            np.ndarray: The Jacobian matrix.
        """
        
        # Compute the derivative of the transformation to standard normal
        gdiff = self.prior.prior_to_normal(x)[1]

        # Scale the precomputed Jacobian by the transformation derivative and noise level
        return (1/self.likelihood.lam)*gdiff*self.jac_const

