"""
Base classes for member of the Exponential Family of distribution (EFD).

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import abc


class EFDStats(object):
    """Wrapper of EFD statistics."""

    def __init__(self, stats):
        """Initialize the list of stats.

        Parameters
        ----------
        stats : list
            List of numpy.ndarray.

        """
        self._stats = stats

    def __getitem__(self, k):
        return self._stats[k]

    def __iadd__(self, stats):
        for i in range(len(self._stats)):
            self._stats[i] += stats[i]
        return self

    def __imul__(self, scale):
        for i in range(len(self._stats)):
            self._stats[i] *= scale
        return self


class EFDPrior(metaclass=abc.ABCMeta):
    """Abstract base class for a prior from the EFD."""

    # Abstract interface.
    # ------------------------------------------------------------------

    @abc.abstractproperty
    def natural_params(self):
        """Vector of natural parameters."""
        pass

    @abc.abstractproperty
    def log_partition(self):
        """Log-partition of the distribution."""
        pass

    @abc.abstractproperty
    def grad_log_partition(self):
        """Gradient of the log-partition.

        Gradient of the log-partition with respect to the natural
        parameters. This corresponds to the expected value of the
        sufficient statistics of the current density/distribution.

        """
        pass

    # ------------------------------------------------------------------

    def kl_div(self, dist):
        """Kullback-Leibler divergence.

        Compute the Kullback-Leibler divergence between the current
        distribution and the given distribution.

        Parameters
        ----------
        dist : EFDPrior
            Distribution of the same type as the current distribution
            (i.e. "self").

        Returns
        -------
        div : float
            Results of the KL divergence in nats.

        """
        # Expected value of the sufficient statistics with respect to
        # the current distribution
        expected_value = self.grad_log_partition

        # Natural parameters of the current and given distributions.
        nparams1 = self.natural_params
        nparams2 = dist.natural_params

        # Log-partition of the current and given distributions.
        log_partition1 = self.log_partition
        log_partition2 = dist.log_partition

        # Compute the KL divergence.
        retval = (nparams1 - nparams2).dot(expected_value)
        retval += log_partition2 - log_partition1

        return retval


class EFDLikelihood(metaclass=abc.ABCMeta):
    """Abstract base class for a likelihood from the EFD."""

    # Abstract interface.
    # ------------------------------------------------------------------

    @abc.abstractstaticmethod
    def get_sufficient_stats(data):
        """Sufficient statistics of the current distribution.

        Parameters
        ----------
        data : numpy.ndarray
            (NxD) matrix where N is the number of frames and D is the
            dimension of a single features vector.

        Returns
        -------
        s_stats : numpy.ndarray
            (NxD2) matrix of sufficient statistics. D2 is the dimension
            of the sufficient statistics for a single features frame.

        """
        pass

    @abc.abstractproperty
    def prior(self):
        """Conjugate prior distribution."""
        pass

    @abc.abstractproperty
    def posterior(self):
        """Conjugate posterior distribution."""
        pass

    def update_posterior(self, acc_s_stats):
        """Update the posterior distribution.

        Parameters
        ----------
        acc_s_stats : numpy.ndarray
            Accumulated sufficient statistics.

        """
        self.posterior.natural_params = self.prior.natural_params + acc_s_stats

    # ------------------------------------------------------------------
