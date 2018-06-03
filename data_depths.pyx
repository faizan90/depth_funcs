# cython: nonecheck=True
# cython: boundscheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)
# cython: embedsignature=True

import numpy as np
cimport numpy as np
from cython.parallel import prange, threadid

from .dtypes cimport DT_D, DT_UL, DT_ULL

DT_D_NP = np.float64
DT_UL_NP = np.uint64
DT_LL_NP = np.int64


cdef extern from "./quick_sort.h" nogil:
    cdef:
        void quick_sort(DT_D arr[], DT_UL first_index, DT_UL last_index)


cdef extern from "./searchsorted.h" nogil:
    cdef:
        DT_UL searchsorted(DT_D arr[], DT_D value, DT_UL arr_size)


cdef inline abs(DT_D ref):
    if ref < 0:
        return -ref
    else:
        return ref


cpdef np.ndarray depth_ftn_mp(
    const DT_D[:, :] ref, 
    const DT_D[:, :] test, 
    const DT_D[:, :] uvecs,
          DT_UL n_cpus=1):

    cdef:
        Py_ssize_t i, j, k
        DT_UL n_mins, n_uvecs, tid, _idx, n_x, n_dims
        DT_D dy_med, _inc_mult = (1 - (1e-7))
        
        DT_UL[:, ::1] mins, numl
        DT_D[:, ::1] ds, dys, dy_sort

    n_x = ref.shape[0]
    n_mins = test.shape[0]
    n_uvecs = uvecs.shape[0]
    n_dims = uvecs.shape[1]
    mins = np.full((n_cpus, n_mins), n_x, dtype=DT_LL_NP)

    ds = np.zeros((n_cpus, n_x), dtype=DT_D_NP)
    dys = np.zeros((n_cpus, n_mins), dtype=DT_D_NP)
    dy_sort = dys.copy()

    numl = np.zeros((n_cpus, n_mins), dtype=DT_LL_NP)

    for i in prange(n_uvecs, 
                    schedule='dynamic', 
                    nogil=True, 
                    num_threads=n_cpus):
        tid = threadid()

        for j in range(n_x):
            ds[tid, j] = 0.0
            for k in range(n_dims):
                ds[tid, j] = ds[tid, j] + (uvecs[i, k] * ref[j, k])

        for j in range(n_mins):
            dys[tid, j] = 0.0
            for k in range(n_dims):
                dys[tid, j] = dys[tid, j] + (uvecs[i, k] * test[j, k])
            dy_sort[tid, j] = dys[tid, j]

        quick_sort(&ds[tid, 0], <DT_UL> 0, <DT_UL> (n_x - 1))
        quick_sort(&dy_sort[tid, 0], <DT_UL> 0, <DT_UL> (n_mins - 1))

        if (n_mins % 2) == 0:
            dy_med = (dy_sort[tid, <DT_UL> (n_mins / 2)] + 
                      dy_sort[tid, <DT_UL> ((n_mins / 2) - 1)]) / 2.0
        else:
            dy_med = dy_sort[tid, <DT_UL> (n_mins / 2)]

        for j in range(n_mins):
            dys[tid, j] = ((dys[tid, j] - dy_med) * _inc_mult) + dy_med
 
        for j in range(n_mins):
            numl[tid, j] = searchsorted(&ds[tid, 0], dys[tid, j], n_x)
 
        for j in range(n_mins):
            _idx = n_x - numl[tid, j]

            if _idx < numl[tid, j]:
                numl[tid, j] = _idx

            if numl[tid, j] < mins[tid, j]:
                mins[tid, j] = numl[tid, j]

    return np.asarray(mins).min(axis=0).astype(DT_UL_NP)
