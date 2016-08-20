
cimport numpy as np
cimport cython
try:
    from sselogsumexp import logsumexp
except Exception:
    from scipy.misc import logsumexp


@cython.boundscheck(False)
@cython.wraparound(False)
def _fast_logsumexp_axis1(np.ndarray[dtype=np.float32_t, ndim=2] array, 
                          np.ndarray[dtype=np.float32_t, ndim=1] out):

    for i in range(array.shape[0]):
        out[i] = _sse_logsumexp(array[i])

