# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)
# cython: embedsignature=True

import numpy as np
cimport numpy as np
from cython.parallel import prange, threadid


DT_D_NP = np.float64
DT_UL_NP = np.uint64
DT_LL_NP = np.int64


cdef extern from "./quick_sort.h" nogil:
    cdef:
        void quick_sort_f64(DT_D *arr, DT_UL first_index, DT_UL last_index)


cdef extern from "./searchsorted.h" nogil:
    cdef:
        DT_UL searchsorted_f64(DT_D *arr, DT_D value, DT_UL arr_size)


cpdef void cmpt_sorted_dot_prods_with_shrink(
        const DT_D[:, ::1] data,
              DT_D[:, ::1] sdp,
              DT_D[:, ::1] shdp,
        const DT_D[:, ::1] uvecs,
              DT_UL n_cpus=1,
        ):

    # sdp = sorted dot product
    # shdp = shrinked sdp

    cdef:
        Py_ssize_t i, j, k, tid

        DT_UL n_pts = data.shape[0]
        DT_UL n_dims = data.shape[1]
        DT_UL n_uvecs = uvecs.shape[0]

        DT_D dy_med, inc_mult = (1 - (1e-10))

    assert n_pts == sdp.shape[1]
    assert n_pts == shdp.shape[1]

    assert n_uvecs == sdp.shape[0]
    assert n_uvecs == shdp.shape[0]
    assert n_dims == uvecs.shape[1]

    assert n_pts > 1
    assert n_uvecs > 0
    assert n_dims > 0
    assert n_cpus > 0

    for i in prange(
        n_uvecs, schedule='dynamic', nogil=True, num_threads=n_cpus):

        tid = threadid()

        for j in range(n_pts):
            shdp[i, j] = 0.0

            for k in range(n_dims):
                shdp[i, j] = shdp[i, j] + (uvecs[i, k] * data[j, k])

            sdp[i, j] = shdp[i, j]

        quick_sort_f64(&sdp[i, 0], <DT_UL> 0, <DT_UL> (n_pts - 1))

        if (n_pts % 2) == 0:
            dy_med = 0.5 * (sdp[tid, n_pts / 2] + sdp[tid, (n_pts / 2) - 1])

        else:
            dy_med = sdp[tid, n_pts / 2]

        for j in range(n_pts):
            shdp[i, j] = ((shdp[i, j] - dy_med) * inc_mult) + dy_med

    return


cpdef np.ndarray get_sdp_depths(
        const DT_D[:, ::1] sdp,
        const DT_D[:, ::1] shdp,
              DT_UL n_cpus=1,
        ):

    # sdp = sorted dot product
    # shdp = shrinked sdp

    cdef:
        Py_ssize_t i, j, tid

        DT_UL dth
        DT_UL n_uvecs = sdp.shape[0]
        DT_UL n_ref_pts = sdp.shape[1]
        DT_UL n_test_pts = sdp.shape[1]

        DT_UL[:, ::1] depths, temp_depths

    assert n_uvecs == shdp.shape[0]

    assert n_ref_pts > 1
    assert n_test_pts > 1
    assert n_uvecs > 0
    assert n_cpus > 0

    depths = np.full((n_cpus, n_test_pts), n_ref_pts, dtype=DT_LL_NP)
    temp_depths = np.empty((n_cpus, n_test_pts), dtype=DT_LL_NP)

    for i in prange(
        n_uvecs, schedule='static', nogil=True, num_threads=n_cpus):

        tid = threadid()

        for j in range(n_test_pts):
            temp_depths[tid, j] = searchsorted_f64(
                &sdp[i, 0], shdp[i, j], n_ref_pts)
 
        for j in range(n_test_pts):
            dth = n_ref_pts - temp_depths[tid, j]

            if dth > temp_depths[tid, j]:
                dth = temp_depths[tid, j] 

            if dth < depths[tid, j]:
                depths[tid, j] = dth

    return np.asarray(depths).min(axis=0).astype(DT_LL_NP)
