# cython: nonecheck=False, boundscheck=False, wraparound=False
# cython: cdivision=True
# cython: language_level=3
import time

import numpy as np
cimport numpy as np

ctypedef unsigned long DT_UL
ctypedef unsigned long long DT_ULL
ctypedef double DT_D
ctypedef np.float64_t DT_D_NP_t
ctypedef np.uint64_t DT_UL_NP_t

DT_D_NP = np.float64


cdef extern from "rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        DT_D rand_c_mp(DT_ULL *rnd_j)
        void warm_up_mp(DT_ULL *seeds_arr, DT_UL n_seeds)  # call this everytime

warm_up()

cdef void test_ftn():
    cdef:
        DT_UL i, n_seeds = 10
        np.ndarray[np.uint64_t, ndim=1, mode='c'] seeds_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] rnds_arr
                
    seeds_arr = np.zeros(n_seeds, dtype=np.uint64)
    rnds_arr = np.zeros(n_seeds, dtype=np.float64)
    
    for i in range(n_seeds):
        seeds_arr[i] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)
    
    print(seeds_arr)
    print(rnds_arr)
    
    warm_up_mp(&seeds_arr[0], n_seeds)
    print(seeds_arr)
    
    for i in range(n_seeds):
        rnds_arr[i] = rand_c_mp(&seeds_arr[i])
        
    print(rnds_arr)

    print(seeds_arr)
    
    for i in range(n_seeds):
        rnds_arr[i] = rand_c_mp(&seeds_arr[i])
        
    print(rnds_arr)
    return
    
# test_ftn()

cpdef void gen_n_rands(DT_ULL n_rands):
    cdef:
        DT_ULL i
     
    for i in range(n_rands):
        rand_c()
         
    return 


cpdef np.ndarray get_n_rands(DT_ULL n_rands):
    cdef:
        DT_ULL i
        DT_D[::1] rnds_arr
        
    rnds_arr = np.zeros(n_rands, dtype=np.float64)
     
    for i in range(n_rands):
        rnds_arr[i] = rand_c()
         
    return np.asarray(rnds_arr)


cpdef void gen_n_rands_mp(DT_ULL n_rands):
    cdef:
        DT_ULL i, n_seeds = 1
        np.ndarray[np.uint64_t, ndim=1, mode='c'] seeds_arr
        np.ndarray[DT_D_NP_t, ndim=1, mode='c'] rnds_arr
                
    seeds_arr = np.zeros(n_seeds, dtype=np.uint64)
    
    for i in range(n_seeds):
        seeds_arr[i] = <DT_ULL> (time.time() * 10000)
        time.sleep(0.001)

    for i in range(n_rands):
        rand_c_mp(&seeds_arr[0])
        
    return
