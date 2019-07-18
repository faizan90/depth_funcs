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


cdef extern from "data_depths_cftns.h" nogil:
    cdef:
        void depth_ftn_c_f64(
            const double *ref,
            const double *test,
            const double *uvecs,
                  double *dot_ref,
                  double *dot_test,
                  double *dot_test_sort,
                  long long *temp_mins,
                  long long *mins,
            const long n_ref,
            const long n_test,
            const long n_uvecs,
            const long n_dims,
            const long n_cpus)

        void depth_ftn_c_f64_no_median(
            const double *ref,
            const double *test,
            const double *uvecs,
                  double *dot_ref,
                  double *dot_test,
                  double *dot_test_sort,
                  long long *temp_mins,
                  long long *mins,
            const long n_ref,
            const long n_test,
            const long n_uvecs,
            const long n_dims,
            const long n_cpus)

        void depth_ftn_c_f32(
            const float *ref,
            const float *test,
            const float *uvecs,
                  float *dot_ref,
                  float *dot_test,
                  float *dot_test_sort,
                  long *temp_mins,
                  long *mins,
            const long n_ref,
            const long n_test,
            const long n_uvecs,
            const long n_dims,
            const long n_cpus)

#         void depth_ftn_c_gf32(
#             const float *ref,
#             const float *test,
#             const float *uvecs,
#                   long *depths,
#             const long n_ref,
#             const long n_test,
#             const long n_uvecs,
#             const long n_dims)


cpdef np.ndarray depth_ftn_mp(
        const DT_D[:, ::1] ref, 
        const DT_D[:, ::1] test, 
        const DT_D[:, ::1] uvecs,
              long n_cpus=1,
              DT_UL depth_type=1) except +:

    cdef:
        Py_ssize_t i, j, k
        long n_mins, n_uvecs, n_ref, n_dims

        long long[:, ::1] mins_i64, temp_mins_i64
        double[:, ::1] dot_ref_f64, dot_test_f64, dot_test_sort_f64

        float[:, ::1] ref_f32, test_f32, uvecs_f32
        long[:, ::1] mins_i32, temp_mins_i32
        float[:, ::1] dot_ref_f32, dot_test_f32, dot_test_sort_f32

        long[::1] gdepths_i32

    assert ref.shape[0] and ref.shape[1], 'No values in ref!'
    assert test.shape[0] and ref.shape[1], 'No values in test!'
    assert uvecs.shape[0] and uvecs.shape[1], 'No values in uvecs!'
    assert ref.shape[1] == test.shape[1] == uvecs.shape[1], (
        'Unequal ref, test and uvecs dimensions!')
    assert n_cpus > 0, 'n_cpus should be more than 0!'

    assert 0 <= depth_type <= 3
    assert depth_type != 2

    n_ref = ref.shape[0]
    n_mins = test.shape[0]
    n_uvecs = uvecs.shape[0]
    n_dims = uvecs.shape[1]

    if depth_type == 0:
        ref_f32 = ref.astype(np.float32)
        test_f32 = test.astype(np.float32)
        uvecs_f32 = uvecs.astype(np.float32)

        mins_i32 = np.full((n_cpus, n_mins), n_ref, dtype=np.int32)
        dot_ref_f32 = np.zeros((n_cpus, n_ref), dtype=np.float32)
        dot_test_f32 = np.zeros((n_cpus, n_mins), dtype=np.float32)
        dot_test_sort_f32 = dot_test_f32.copy()

        temp_mins_i32 = np.zeros((n_cpus, n_mins), dtype=np.int32)

        depth_ftn_c_f32(
            &ref_f32[0, 0],
            &test_f32[0, 0],
            &uvecs_f32[0, 0],
            &dot_ref_f32[0, 0],
            &dot_test_f32[0, 0],
            &dot_test_sort_f32[0, 0],
            &temp_mins_i32[0, 0],
            &mins_i32[0, 0],
            n_ref,
            n_mins,
            n_uvecs,
            n_dims,
            n_cpus)
        return np.asarray(mins_i32).min(axis=0).astype(np.int32)

    elif (depth_type == 1) or (depth_type == 3):
        mins_i64 = np.full((n_cpus, n_mins), n_ref, dtype=DT_LL_NP)
        temp_mins_i64 = np.zeros((n_cpus, n_mins), dtype=DT_LL_NP)

        dot_ref_f64 = np.zeros((n_cpus, n_ref), dtype=DT_D_NP)
        dot_test_f64 = np.zeros((n_cpus, n_mins), dtype=DT_D_NP)
        dot_test_sort_f64 = dot_test_f64.copy()
        
        if depth_type == 1:

            depth_ftn_c_f64(
                &ref[0, 0],
                &test[0, 0],
                &uvecs[0, 0],
                &dot_ref_f64[0, 0],
                &dot_test_f64[0, 0],
                &dot_test_sort_f64[0, 0],
                &temp_mins_i64[0, 0],
                &mins_i64[0, 0],
                n_ref,
                n_mins,
                n_uvecs,
                n_dims,
                n_cpus)
        else:
            depth_ftn_c_f64_no_median(
                &ref[0, 0],
                &test[0, 0],
                &uvecs[0, 0],
                &dot_ref_f64[0, 0],
                &dot_test_f64[0, 0],
                &dot_test_sort_f64[0, 0],
                &temp_mins_i64[0, 0],
                &mins_i64[0, 0],
                n_ref,
                n_mins,
                n_uvecs,
                n_dims,
                n_cpus)

        return np.asarray(mins_i64).min(axis=0).astype(DT_LL_NP)

#     elif depth_type == 2:
#         ref_f32 = ref.astype(np.float32)
#         test_f32 = test.astype(np.float32)
#         uvecs_f32 = uvecs.astype(np.float32)
#
#         gdepths_i32 = np.full(n_mins, n_ref, dtype=np.int32)
# 
#         depth_ftn_c_gf32(
#             &ref_f32[0, 0],
#             &test_f32[0, 0],
#             &uvecs_f32[0, 0],
#             &gdepths_i32[0],
#             n_ref,
#             n_mins,
#             n_uvecs,
#             n_dims)
#         [print(gdepths_i32[i]) for i in range(n_mins)]
#         return np.asarray(gdepths_i32).astype(np.int32)


cpdef np.ndarray depth_ftn_mp_v2(
        const DT_D[:, ::1] ref, 
        const DT_D[:, ::1] test, 
        const DT_D[:, ::1] uvecs,
              DT_UL n_cpus=1):
          
    '''Keep this one as a benchmark
    '''

    cdef:
        Py_ssize_t i, j, k
        DT_UL n_mins, n_uvecs, tid, _idx, n_x, n_dims
        DT_D dy_med, _inc_mult = (1 - (1e-10))

#         DT_UL[::1] zero_d_arr
        DT_UL[:, ::1] mins, numl
        DT_D[:, ::1] ds, dys, dys_sort

    assert ref.shape[0] and ref.shape[1], 'No values in ref!'
    assert test.shape[0] and ref.shape[1], 'No values in test!'
    assert uvecs.shape[0] and uvecs.shape[1], 'No values in uvecs!'
    assert ref.shape[1] == test.shape[1] == uvecs.shape[1], (
        'Unequal ref, test and uvecs dimensions!')
    assert n_cpus > 0, 'n_cpus should be more than 0!'

    n_x = ref.shape[0]
    n_mins = test.shape[0]
    n_uvecs = uvecs.shape[0]
    n_dims = uvecs.shape[1]
    mins = np.full((n_cpus, n_mins), n_x, dtype=DT_LL_NP)
#     zero_d_arr = np.zeros(n_mins, dtype=DT_LL_NP)

    ds = np.zeros((n_cpus, n_x), dtype=DT_D_NP)
    dys = np.zeros((n_cpus, n_mins), dtype=DT_D_NP)
    dys_sort = dys.copy()

    numl = np.zeros((n_cpus, n_mins), dtype=DT_LL_NP)

    for i in prange(
        n_uvecs, schedule='dynamic', nogil=True, num_threads=n_cpus):

        tid = threadid()

        for j in range(n_x):
            ds[tid, j] = 0.0
            for k in range(n_dims):
                ds[tid, j] = ds[tid, j] + (uvecs[i, k] * ref[j, k])

        for j in range(n_mins):
#             if zero_d_arr[j]:
#                 continue

            dys[tid, j] = 0.0
            for k in range(n_dims):
                dys[tid, j] = dys[tid, j] + (uvecs[i, k] * test[j, k])
            dys_sort[tid, j] = dys[tid, j]

        quick_sort_f64(&ds[tid, 0], <DT_UL> 0, <DT_UL> (n_x - 1))
        quick_sort_f64(&dys_sort[tid, 0], <DT_UL> 0, <DT_UL> (n_mins - 1))

        if (n_mins % 2) == 0:
            dy_med = 0.5 * (
                dys_sort[tid, n_mins / 2] + dys_sort[tid, (n_mins / 2) - 1])

        else:
            dy_med = dys_sort[tid, n_mins / 2]

        for j in range(n_mins):
            dys[tid, j] = ((dys[tid, j] - dy_med) * _inc_mult) + dy_med

        for j in range(n_mins):
#             if zero_d_arr[j]:
#                 continue

            numl[tid, j] = searchsorted_f64(&ds[tid, 0], dys[tid, j], n_x)
 
        for j in range(n_mins):
#             if zero_d_arr[j]:
#                 mins[tid, j] = 0
#                 continue

            _idx = n_x - numl[tid, j]

            if _idx < numl[tid, j]:
                numl[tid, j] = _idx

            if numl[tid, j] < mins[tid, j]:
                mins[tid, j] = numl[tid, j]

#                 # lets hope, threads dont write to at the sametime
#                 if mins[tid, j] == 0:
#                     zero_d_arr[j] = 1

    return np.asarray(mins).min(axis=0).astype(DT_LL_NP)
