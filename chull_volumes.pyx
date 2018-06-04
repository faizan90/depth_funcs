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

        DT_UL n_in_pts, n_dims, iter_ct, re_seed_i = <DT_UL> 1e6
        DT_ULL n_inside_pts, n_outside_pts, n_tot_pts
        DT_D pre_rel, curr_rel, curr_vol = np.nan, tot_vol, rn_ct = 0.0

        DT_ULL[::1] depths_arr
        DT_D[::1] diffs_arr
        DT_D[:, ::1] crds_min_max_arr, rand_pts_arr

    assert in_chull_pts.shape[0], 'No points in hull!'
    assert in_chull_pts.shape[1], 'No dimensions in hull!'
    assert uvecs.shape[0], 'No unit vectors!'
    assert uvecs.shape[1], 'No dimnsions in unit vectors!'
    assert in_chull_pts.shape[1] == uvecs.shape[1], (
        'Hull and unit vector dimensions unequal!')
    assert chk_iter > 0, 'chk_iter should be more than 0!'
    assert max_iters > 0, 'max_iters should be more than 0!'
    assert n_cpus > 0, 'n_cpus should be greater than 0!'
    assert 0 < vol_tol < 1, 'vol_tol should be in between 0 and 1!'

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
        if (rn_ct / re_seed_i) > 1.0:
            re_seed()
            rn_ct = 0.0

        for i in range(chk_iter):
            for j in range(n_dims):
                rand_pts_arr[i, j] = (
                    crds_min_max_arr[j, 0] + (diffs_arr[j] * rand_c()))
        rn_ct += (chk_iter * n_dims)

        depths_arr = np.asarray(
            depth_ftn_mp(in_chull_pts, rand_pts_arr, uvecs, n_cpus))

        for i in range(chk_iter):
            if depths_arr[i] == 0:
                n_outside_pts += 1
            else:
                n_inside_pts += 1
            n_tot_pts += 1

        if n_tot_pts and n_outside_pts: 
            curr_vol = n_inside_pts / (<DT_D> n_tot_pts)
            curr_rel = n_inside_pts / (<DT_D> n_outside_pts)
            if ((abs(curr_rel - pre_rel) / pre_rel) < vol_tol) and (iter_ct >= 0):
                break

            print('curr_vol:', curr_vol,
                  'n_inside_pts:', n_inside_pts,
                  'n_outside_pts:', n_outside_pts,
                  'n_tot_pts:', n_tot_pts,
                  'curr_rel:', curr_rel)
            pre_rel = curr_rel
        iter_ct += 1
        break

    tot_vol = curr_vol
    for i in range(n_dims):
        tot_vol *= diffs_arr[i]
    return abs(tot_vol)
