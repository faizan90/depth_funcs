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

from .data_depths cimport depth_ftn_mp

DT_D_NP = np.float64
DT_UL_NP = np.uint64
DT_LL_NP = np.int64


cdef extern from "./rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        void re_seed()


warm_up()


cpdef DT_D cmpt_rand_pts_chull_vol(
    const DT_D[:, :] in_chull_pts,
    const DT_D[:, :] uvecs,
          DT_UL chk_iter,
          DT_UL max_iters, 
          DT_UL n_cpus=1,
          DT_D vol_tol=0.01):

    cdef:
        Py_ssize_t i, j

        DT_UL n_in_pts, n_dims, iter_ct
        DT_ULL n_inside_pts, n_outside_pts, n_tot_pts
        DT_D pre_rel, curr_rel, curr_vol, tot_vol

        DT_ULL[::1] depths_arr
        DT_D[::1] diffs_arr
        DT_D[:, ::1] crds_min_max_arr, rand_pts_arr

    n_in_pts = in_chull_pts.shape[0]
    n_dims = in_chull_pts.shape[1]

    crds_min_max_arr = np.empty((n_dims, 2), dtype=DT_D_NP)
    diffs_arr = np.empty(n_dims, dtype=DT_D_NP)

    for i in range(n_dims):
        crds_min_max_arr[i, 0] = np.asarray(in_chull_pts[:, i]).min()
        crds_min_max_arr[i, 1] = np.asarray(in_chull_pts[:, i]).max()
        diffs_arr[i] = <DT_D> (crds_min_max_arr[i, 1] - crds_min_max_arr[i, 0])

#     raise Exception('x')
    rand_pts_arr = np.empty((chk_iter, n_dims), dtype=DT_D_NP)
 
    iter_ct = 0
    n_inside_pts = 0
    n_outside_pts = 0
    n_tot_pts = 0
    pre_rel = 1
    curr_rel = 0
    while (iter_ct < max_iters):
        re_seed()
        for i in range(chk_iter):
            for j in range(n_dims):
                rand_pts_arr[i, j] = crds_min_max_arr[j, 0] + (diffs_arr[j] * rand_c())

        depths_arr = np.asarray(depth_ftn_mp(in_chull_pts, rand_pts_arr, uvecs, n_cpus))

        for i in range(chk_iter):
            if depths_arr[i] == 0:
                n_outside_pts += 1
            else:
                n_inside_pts += 1
            n_tot_pts += 1

        curr_vol = n_inside_pts / (<DT_D> n_tot_pts)
        curr_rel = n_inside_pts / (<DT_D> n_outside_pts)
        if ((abs(curr_rel - pre_rel) / pre_rel) < vol_tol) and (iter_ct > 3):
            break

        print('curr_vol:', curr_vol, 'n_inside_pts:', n_inside_pts, 'n_outside_pts:', n_outside_pts, 'n_tot_pts:', n_tot_pts, 'curr_rel:', curr_rel)
        pre_rel = curr_rel
        iter_ct += 1

    tot_vol = curr_vol
    for i in range(n_dims):
        tot_vol *= diffs_arr[i]
    return abs(tot_vol)
