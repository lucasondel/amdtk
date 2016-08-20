
cimport numpy as np
cimport cython
from libc.math cimport isnan
try:
    from sselogsumexp import logsumexp
except Exception:
    from scipy.misc import logsumexp

cdef np.float32_t neg_inf = float('-inf')

@cython.boundscheck(False)
@cython.wraparound(False)
def _fast_logsumexp_axis1(np.ndarray[dtype=np.float32_t, ndim=2] array, 
                          np.ndarray[dtype=np.float32_t, ndim=1] out):

    for i in range(array.shape[0]):
        out[i] = logsumexp(array[i])
        if isnan(out[i]):
            out[i] = neg_inf

