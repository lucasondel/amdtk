"""
Prior for the Structured Variational Auto-Encoder.

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


class SVAEPrior(metaclass=abc.ABCMeta):
    """Interface for the SVAE prior."""

    @abc.abstractmethod
    def init_resps(self, n_frames):
        """Get the initialize per-frame responsibilities.

        Parameters
        ----------
        n_frames : numpy.ndarray,
            Number of frames

        Returns
        -------
        resps : numpy.ndarray
            Initial per-frame responsibilities.

        """
        pass

    @abc.abstractmethod
    def get_resps(self, s_stats):
        """Per-frame responsibilities.

        Parameters
        ----------
        s_stats : numpy.ndarray,
            Sufficient statistics.

        Returns
        -------
        resps : numpy.ndarray
            Per-frame responsibilities.

        """
        pass

    @abc.abstractmethod
    def accumulate_stats(self, s_stats, resps):
        """Accumulate the sufficient statistics.

        Parameters
        ----------
        s_stats : numpy.ndarray
            Sufficient statistics.
        resps : numpy.ndarray
            Per-frame responsibilities.

        Returns
        -------
        acc_stats : :class:`EFDStats`
            Accumulated sufficient statistics.

        """
        pass
