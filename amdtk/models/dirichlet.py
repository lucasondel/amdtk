
"""Dirichlet prior for mixture models."""

from scipy.special import gammaln, psi


class Dirichlet(object):
    """Dirichlet density."""

    def __init__(self, alphas):
        self.alphas = alphas

    def expected_log_weights(self):
        """Expected value of the logarithm of the weights.

        Returns
        -------
        E_log_pi : numpy.ndarray
            Log weights.

        """
        return psi(self.alphas) - psi(self.alphas.sum())

    def kl_divergence(self, pdf):
        '''KL divergence between the current and the given densities.

        pdf : :class:`Dirichlet`
            Dirichlet density to compute the divergence with.

        Returns
        -------
        KL : float
            KL divergence.

        '''
        exp_log_weights = self.expected_log_weights()
        kl_div = gammaln(self.alphas.sum())
        kl_div -= gammaln(pdf.alphas.sum())
        kl_div -= gammaln(self.alphas).sum()
        kl_div += gammaln(pdf.alphas).sum()
        kl_div += (exp_log_weights*(self.alphas - pdf.alphas)).sum()
        return kl_div

    def new_posterior(self, stats):
        """Create a new posterior distribution given the parameters of the
        current model and the statistics provided.

        Parameters
        ----------
        stats : :class:MultivariateGaussianDiagCovStats
            Accumulated sufficient statistics for the update.

        Returns
        -------
        post : :class:Dirichlet
            New Dirichlet density.

        """
        return Dirichlet(self.alphas + stats[0])
