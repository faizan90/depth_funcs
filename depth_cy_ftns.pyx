# cython: nonecheck=False, boundscheck=False, wraparound=False
# cython: cdivision=True
# cython: language_level=3

import time
import os

import numpy as np
cimport numpy as np
from cython.parallel import prange, threadid

ctypedef double DT_D
ctypedef long long DT_UL
ctypedef unsigned long long DT_ULL

ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t
ctypedef np.int64_t DT_LL_NP_t


cdef extern from "rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        void re_seed()
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime

warm_up()

cdef extern from "quick_sort.h" nogil:
    cdef:
        void quick_sort(DT_D arr[],
                        DT_UL first_index,
                        DT_UL last_index)


cpdef np.ndarray gen_usph_vecs_mp(DT_UL n_vecs, DT_UL n_dims, DT_UL n_cpus):
    cdef:
        Py_ssize_t i, j
        DT_UL tid, re_seed_i = int(1e6)
        DT_D curr_rand, mag
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] vec, vecs_arr
        np.ndarray[DT_UL_NP_t, ndim=1, mode='c'] seeds_arr

    vec = np.zeros((n_cpus, n_dims), dtype=np.float64)
    mags_arr = np.zeros(n_vecs, dtype=np.float64)
    vecs_arr = np.zeros((n_vecs, n_dims), dtype=np.float64)
    seeds_arr = np.zeros(n_cpus, dtype=np.uint64)
            
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

        for j in range(n_dims):
            vecs_arr[i, j] = vec[tid, j] / mag

        if (i % (re_seed_i + tid)) == 0:
            with gil:
                seeds_arr[tid] = <DT_ULL> (time.time() * 10000)
                time.sleep(0.001)
               
            for j in range(1000):
                rand_c_mp(&seeds_arr[tid])

    return vecs_arr


cpdef np.ndarray gen_usph_vecs(n_vecs, n_dims):
    cdef:
        Py_ssize_t i, j
        DT_UL re_seed_i = int(1e6)
        DT_D curr_rand, mag
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] vec
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] vecs_arr

    vec = np.zeros(n_dims, dtype=np.float64)
    mags_arr = np.zeros(n_vecs, dtype=np.float64)
    vecs_arr = np.zeros((n_vecs, n_dims), dtype=np.float64)

    for i in range(n_vecs):
        mag = 2.0
        while mag > 1.0:
            mag = 0.0
            for j in range(n_dims):
                curr_rand = -1 + (2 * rand_c())
                vec[j] = curr_rand
                mag += curr_rand**2
            mag = mag**0.5

        for j in range(n_dims):
            vecs_arr[i, j] = vec[j] / mag

        if (i % re_seed_i) == 0:
            re_seed()
    return vecs_arr


cpdef np.ndarray depth_ftn_mp(const DT_D_NP_t[:, :] x, 
                              const DT_D_NP_t[:, :] y, 
                              const DT_D_NP_t[:, :] ei,
                              DT_UL n_cpus=1):
    
    cdef:
        Py_ssize_t i, j, k
        DT_UL n_mins, n_ei, tid, _idx, n_x, n_dims
        DT_D dy_med, _inc_mult = (1 - (1e-7))
        np.ndarray[DT_LL_NP_t, ndim=2, mode='c'] mins, numl
        np.ndarray[DT_D_NP_t, ndim=2, mode='c'] ds, dys, dy_sort

    n_x = x.shape[0]
    n_mins = y.shape[0]
    n_ei = ei.shape[0]
    n_dims = ei.shape[1]
    mins = np.full((n_cpus, n_mins), n_x, dtype=np.int64)

    ds = np.zeros((n_cpus, x.shape[0]), dtype=np.float64)
    dys = np.zeros((n_cpus, y.shape[0]), dtype=np.float64)
    dy_sort = dys.copy()
    
    numl = np.zeros((n_cpus, n_mins), dtype=np.int64)

#     os.environ['MKL_NUM_THREADS'] = str(1)
#     os.environ['NUMEXPR_NUM_THREADS'] = str(1)
#     os.environ['OMP_NUM_THREADS'] = str(1)
#
#     np.dot(ei, x.T, out=ds)
#     np.dot(ei, y.T, out=dys)
#     dy_sort = dys.copy()
#         
    for i in prange(n_ei, schedule='static', nogil=True, num_threads=n_cpus):
        tid = threadid()
        
        for j in range(n_x):
            ds[tid, j] = 0.0
            for k in range(n_dims):
                ds[tid, j] = ds[tid, j] + (ei[i, k] * x[j, k])

        for j in range(n_mins):
            dys[tid, j] = 0.0
            for k in range(n_dims):
                dys[tid, j] = dys[tid, j] + (ei[i, k] * y[j, k])
            dy_sort[tid, j] = dys[tid, j]
                
        quick_sort(&ds[tid, 0], <DT_UL> 0, <DT_UL> (n_x - 1))
        quick_sort(&dy_sort[tid, 0], <DT_UL> 0, <DT_UL> (n_mins - 1))
        
        if (n_mins % 2) == 0:
            dy_med = (dy_sort[tid, n_mins / 2] + 
                      dy_sort[tid, n_mins / 2 - 1]) / 2.0
        else:
            dy_med = dy_sort[tid, n_mins / 2]
        
        for j in range(n_mins):
            dys[tid, j] = ((dys[tid, j] - dy_med) * _inc_mult) + dy_med
 
        with gil:
            numl[tid, :] = np.searchsorted(ds[tid, :], dys[tid, :])
            
        for j in range(n_mins):
            _idx = n_mins - numl[tid, j] # 0.0 secs
            
            if _idx < numl[tid, j]:
                numl[tid, j] = _idx
                  
            if numl[tid, j] < mins[tid, j]:
                mins[tid, j] = numl[tid, j]
                  
            if mins[tid, j] < 0:
                mins[tid, j] = 0

    return mins.min(axis=0).astype(np.uint64)
