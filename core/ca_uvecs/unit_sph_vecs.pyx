# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)
# cython: embedsignature=True

import time

import numpy as np
cimport numpy as np
from cython.parallel import prange, threadid


DT_D_NP = np.float64
DT_UL_NP = np.uint64
DT_LL_NP = np.int64


cdef extern from "math.h" nogil:
    cdef DT_D log(DT_D x)
    cdef DT_D M_PI
    cdef DT_D INFINITY


cdef extern from "../ba_rands/rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        void re_seed()
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime


cdef extern from "./usph_vecs.h" nogil:
    cdef:
        void gen_usph_vecs_norm_dist_c(
                unsigned long long *seeds_arr,
                double *rn_ct_arr,
                double *ndim_usph_vecs,
                long n_vecs,
                long n_dims,
                long n_cpus)

warm_up()


cdef DT_D usph_norm_ppf(DT_D p) nogil:
    cdef:
        DT_D t, z

    if p <= 0.0:
        return -INFINITY
    elif p >= 1.0:
        return INFINITY

    if p > 0.5:
        t = (-2.0 * log(1 - p))**0.5
    else:
        t = (-2.0 * log(p))**0.5

    z = -0.322232431088 + t * (-1.0 + t * (-0.342242088547 + t * (
        (-0.020423120245 + t * -0.453642210148e-4))))
    z = z / (0.0993484626060 + t * (0.588581570495 + t * (
        (0.531103462366 + t * (0.103537752850 + t * 0.3856070063e-2)))))
    z = z + t

    if p < 0.5:
        z = -z
    return z


cpdef np.ndarray gen_usph_vecs_norm_dist(
    DT_UL n_vecs,
    DT_UL n_dims,
    DT_UL n_cpus):

    cdef:
        Py_ssize_t i, j
        DT_UL re_seed_i = <DT_UL> 1e6
        DT_D mag, rn_ct = 0.0
        DT_D[:, ::1] ndim_sph_vecs

    assert n_vecs > 0, 'n_vecs should be more than 0!'
    assert n_dims > 0, 'n_dims should be more than 0!'

    ndim_sph_vecs = np.empty((n_vecs, n_dims), dtype=DT_D_NP)

    for i in range(n_vecs):
        mag = 0.0
        for j in range(n_dims):
            ndim_sph_vecs[i, j] = usph_norm_ppf(rand_c())
            mag = mag + (ndim_sph_vecs[i, j]**2)
        mag = mag**0.5

        for j in range(n_dims):
            ndim_sph_vecs[i, j] = ndim_sph_vecs[i, j] / mag

        rn_ct += n_dims
        if (rn_ct / re_seed_i) > 1:
            re_seed()
            rn_ct = 0.0
    return np.asarray(ndim_sph_vecs)


cpdef np.ndarray gen_usph_vecs_norm_dist_mp(
    DT_UL n_vecs,
    DT_UL n_dims,
    DT_UL n_cpus):

    cdef:
        Py_ssize_t i, j, tid
        DT_UL re_seed_i = <DT_UL> 1e6
        DT_D mag

        DT_ULL[::1] seeds_arr
        DT_D[::1] rn_ct_arr
        DT_D[:, ::1] ndim_usph_vecs

    assert n_vecs > 0, 'n_vecs should be more than 0!'
    assert n_dims > 0, 'n_dims should be more than 0!'
    assert n_cpus > 0, 'n_cpus should be more than 0!'

    ndim_usph_vecs = np.empty((n_vecs, n_dims), dtype=DT_D_NP)
    seeds_arr = np.zeros(n_cpus, dtype=DT_UL_NP)
    rn_ct_arr = np.zeros(n_cpus, dtype=DT_D_NP)

    for tid in range(n_cpus):
        seeds_arr[tid] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)
    warm_up_mp(&seeds_arr[0], n_cpus)

    gen_usph_vecs_norm_dist_c(
            &seeds_arr[0],
            &rn_ct_arr[0],
            &ndim_usph_vecs[0, 0],
            n_vecs,
            n_dims,
            n_cpus)

#     for i in prange(n_vecs, schedule='static', nogil=True, num_threads=n_cpus):
#         tid = threadid()
#         mag = 0.0
#         for j in range(n_dims):
#             ndim_usph_vecs[i, j] = usph_norm_ppf(rand_c_mp(&seeds_arr[tid]))
#             mag = mag + (ndim_usph_vecs[i, j]**2)
#         mag = mag**0.5
#  
#         for j in range(n_dims):
#             ndim_usph_vecs[i, j] = ndim_usph_vecs[i, j] / mag
#  
#         rn_ct_arr[tid] = rn_ct_arr[tid] + n_dims
#         if (rn_ct_arr[tid]  / re_seed_i) > 1:
#             with gil:
#                 seeds_arr[tid] = <DT_ULL> (time.time() * 10000)
#                 time.sleep(0.001)
#  
#             for j in range(1000):
#                 rand_c_mp(&seeds_arr[tid])
#             rn_ct_arr[tid] = 0.0
    return np.asarray(ndim_usph_vecs)


cpdef np.ndarray gen_usph_vecs(DT_UL n_vecs, 
                               DT_UL n_dims, 
                               DT_UL n_cpus):
    cdef:
        Py_ssize_t i, j
        DT_UL re_seed_i = <DT_UL> 1e6
        DT_D curr_rand, mag, rn_ct = 0.0

        DT_D[::1] vec
        DT_D[:, ::1] vecs_arr

    assert n_vecs > 0, 'n_vecs should be more than 0!'
    assert n_dims > 0, 'n_dims should be more than 0!'

    vec = np.zeros(n_dims, dtype=DT_D_NP)
    vecs_arr = np.zeros((n_vecs, n_dims), dtype=DT_D_NP)

    for i in range(n_vecs):
        mag = 2.0
        while mag > 1.0:
            mag = 0.0
            for j in range(n_dims):
                curr_rand = -1 + (2 * rand_c())
                vec[j] = curr_rand
                mag += curr_rand**2
            mag = mag**0.5

            rn_ct += n_dims
            if (rn_ct / re_seed_i) > 1:
                re_seed()
                rn_ct = 0.0

        for j in range(n_dims):
            vecs_arr[i, j] = vec[j] / mag

    return np.asarray(vecs_arr)


cpdef np.ndarray gen_usph_vecs_mp(DT_UL n_vecs, DT_UL n_dims, DT_UL n_cpus):
    cdef:
        Py_ssize_t i, j
        DT_UL tid, re_seed_i = <DT_UL> 1e6
        DT_D curr_rand, mag

        DT_ULL[::1] seeds_arr
        DT_D[::1] rn_ct_arr
        DT_D[:, ::1] vec, vecs_arr

    assert n_vecs > 0, 'n_vecs should be more than 0!'
    assert n_dims > 0, 'n_dims should be more than 0!'
    assert n_cpus > 0, 'n_cpus should be more than 0!'

    vec = np.zeros((n_cpus, n_dims), dtype=DT_D_NP)
    vecs_arr = np.zeros((n_vecs, n_dims), dtype=DT_D_NP)
    seeds_arr = np.zeros(n_cpus, dtype=DT_UL_NP)
    rn_ct_arr = np.zeros(n_cpus, dtype=DT_D_NP)

    # prep the MP RNG
    for tid in range(n_cpus):
        seeds_arr[tid] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)
    warm_up_mp(&seeds_arr[0], n_cpus)

    for i in prange(n_vecs, schedule='static', nogil=True, num_threads=n_cpus):
        tid = threadid()
        mag = 2.0
        while mag > 1.0:
            mag = 0.0
            for j in range(n_dims):
                curr_rand = -1 + (2 * rand_c_mp(&seeds_arr[tid]))
                vec[tid, j] = curr_rand
                mag = mag + curr_rand**2
            mag = mag**0.5

            rn_ct_arr[tid] = rn_ct_arr[tid] + n_dims
            if (rn_ct_arr[tid]  / re_seed_i) > 1:
                with gil:
                    seeds_arr[tid] = <DT_ULL> (time.time() * 10000)
                    time.sleep(0.001)

                for j in range(1000):
                    rand_c_mp(&seeds_arr[tid])
                rn_ct_arr[tid] = 0.0

        for j in range(n_dims):
            vecs_arr[i, j] = vec[tid, j] / mag

    return np.asarray(vecs_arr)
