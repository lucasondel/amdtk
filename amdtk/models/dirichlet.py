
"""Dirichlet density."""

from scipy.special import gammaln, psi


class Dirichlet(object):
    """Dirichlet density.

    Attributes
    ----------
    alphas : numpy.ndarray
        Parameters of the Dirichlet density.

    Methods
    -------
    expLogPi()
        Expected value of the logarithm of the weights.
    KL(pdf)
        KL divergence between the current and the given densities.
    newPosterior(stats)
        New posterior distribution.

    """

    def __init__(self, alphas):
        self.alphas = alphas

    def expLogPi(self):
        """Expected value of the logarithm of the weights.

        Returns
        -------
        E_log_pi : numpy.ndarray
            Log weights.

        """
        return psi(self.alphas) - psi(self.alphas.sum())

    def KL(self, pdf):
        '''KL divergence between the current and the given densities.

        Returns
        -------
        KL : float
            KL divergence.

        '''
        E_log_weights = self.expLogPi()
        dirichlet_KL = gammaln(self.alphas.sum())
        dirichlet_KL -= gammaln(pdf.alphas.sum())
        dirichlet_KL -= gammaln(self.alphas).sum()
        dirichlet_KL += gammaln(pdf.alphas).sum()
        dirichlet_KL += (E_log_weights*(self.alphas - pdf.alphas)).sum()
        return dirichlet_KL

    def newPosterior(self, stats):
        """Create a new posterior distribution.

        Create a new Dirichlet density given the parameters of the current
        model and the statistics provided.

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
