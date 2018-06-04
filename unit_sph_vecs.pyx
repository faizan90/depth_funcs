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


cdef extern from "./rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        void re_seed()
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime


warm_up()


cpdef np.ndarray gen_usph_vecs(n_vecs, n_dims):
    cdef:
        Py_ssize_t i, j
        DT_UL re_seed_i = <DT_UL> 1e6
        DT_D curr_rand, mag, rn_ct = 0.0

        DT_D[::1] vec
        DT_D[:, ::1] vecs_arr

    assert n_vecs > 0, 'n_vecs should be more than 0!'
    assert n_dims > 0, 'n_dims should be more than 0!'

    vec = np.zeros(n_dims, dtype=DT_D_NP)
    mags_arr = np.zeros(n_vecs, dtype=DT_D_NP)
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
    mags_arr = np.zeros(n_vecs, dtype=DT_D_NP)
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
