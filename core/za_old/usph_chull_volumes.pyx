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

from .data_depths cimport depth_ftn_mp_v2 as depth_ftn_mp
from .unit_sph_vecs cimport gen_usph_vecs_mp as gen_usph_vecs_norm_dist

DT_D_NP = np.float64
DT_UL_NP = np.uint64
DT_LL_NP = np.int64


cdef extern from "math.h" nogil:
    cdef DT_D M_PI


cdef extern from "./rand_gen_mp.h" nogil:
    cdef:
        DT_D rand_c()
        void warm_up()  # call this everytime
        void re_seed()


warm_up()

cdef DT_UL _fac(DT_UL n):
    if n == 1:
        return n
    else:
        return n * _fac(n - 1)


cdef DT_D _sph_vol_2k(DT_UL *d, DT_D *r):
    cdef:
        DT_UL k = d[0] / 2
        DT_D vol = (M_PI ** k) * (r[0] ** d[0]) / _fac(k)
    return vol


cdef DT_D _sph_vol_2kp1(DT_UL *d, DT_D *r):
    cdef:
        DT_UL k = (d[0] - 1) / 2
        DT_D vol = ((2 * _fac(k)) * 
                    ((4 * M_PI) ** k) * 
                    (r[0] ** d[0]) / 
                    (_fac(d[0])))
    return vol


cdef DT_D get_sph_vol(DT_UL *d, DT_D *r):
    if d[0] % 2 == 0:
        return _sph_vol_2k(d, r)
    else:
        return _sph_vol_2kp1(d, r)


cpdef cmpt_usph_chull_vol(
    const DT_D[:, ::1] in_chull_pts,
    const DT_D[:, ::1] uvecs,
          DT_UL chk_iter,
          DT_UL max_iters, 
          DT_UL n_cpus=1,
          DT_D vol_tol=0.01):

    cdef:
        Py_ssize_t i, j

        DT_UL n_in_pts, n_dims, iter_ct, re_seed_i = <DT_UL> 1e6
        DT_ULL n_inside_pts, n_outside_pts, n_tot_pts
        DT_D pre_rel, curr_rel, curr_vol = np.nan, tot_vol = 0.0, rn_ct = 0.0
        DT_D mag, min_mag = np.inf, max_mag = -np.inf, rnd_mag

        DT_ULL[::1] depths_arr
        DT_D[:, ::1] rand_pts_arr, shif_chull_pts
        DT_D[:, ::1] chull_resc_arr, rnd_uvecs

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

    shif_chull_pts = np.empty((n_in_pts, n_dims), dtype=DT_D_NP)
    crds_min_max_arr = np.empty((n_dims, 2), dtype=DT_D_NP)
    diffs_arr = np.empty(n_dims, dtype=DT_D_NP)

    for i in range(n_dims):
        crds_min_max_arr[i, 0] = np.asarray(in_chull_pts[:, i]).min()
        crds_min_max_arr[i, 1] = np.asarray(in_chull_pts[:, i]).max()
        diffs_arr[i] = <DT_D> (crds_min_max_arr[i, 1] - crds_min_max_arr[i, 0])

    # move to origin
    for i in range(n_in_pts):
        for j in range(n_dims):
            shif_chull_pts[i, j] = (in_chull_pts[i, j] - 
                                    ((crds_min_max_arr[j, 1] + 
                                      crds_min_max_arr[j, 0]) * 0.5))

    for i in range(n_in_pts):
        mag = 0.0
        for j in range(n_dims):
            mag = mag + (shif_chull_pts[i, j]**2)
        mag = mag**0.5

        if mag < min_mag:
            min_mag = mag

        if mag > max_mag:
            max_mag = mag

#         print('%d: %0.5f' % (i, mag))

    min_max_mag_diff = max_mag - min_mag
    print('min_mag: %0.5f, max_mag: %0.5f' % (min_mag, max_mag))

    min_sph_vol = get_sph_vol(&n_dims, &min_mag)
    max_sph_vol = get_sph_vol(&n_dims, &max_mag)
    print('min_sph_vol: %0.5f, max_sph_vol: %0.5f, diff: %0.5f' % 
          (min_sph_vol, max_sph_vol, max_sph_vol - min_sph_vol))
#     raise Exception('Stop!')
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

        rnd_uvecs = gen_usph_vecs_norm_dist(chk_iter, n_dims, n_cpus)
        for i in range(chk_iter):
            rnd_mag = max_mag * (rand_c()**(1.0 / <DT_D> n_dims))
            for j in range(n_dims):
                rand_pts_arr[i, j] = rnd_mag * rnd_uvecs[i, j]
            
        rn_ct += (chk_iter * n_dims)
 
        depths_arr = np.asarray(
            depth_ftn_mp(shif_chull_pts, rand_pts_arr, uvecs, n_cpus))
        
 
        for i in range(chk_iter):
            if depths_arr[i] == 0:
                n_outside_pts += 1
            else:
                n_inside_pts += 1
            n_tot_pts += 1

        if n_tot_pts and n_outside_pts: 
            curr_vol = n_inside_pts / (<DT_D> n_tot_pts)
            curr_rel = n_inside_pts / (<DT_D> n_outside_pts)
            if ((abs(curr_rel - pre_rel) / pre_rel) < vol_tol) and (iter_ct >= 5):
                break
 
            print('curr_vol:', curr_vol,
                  'n_inside_pts:', n_inside_pts,
                  'n_outside_pts:', n_outside_pts,
                  'n_tot_pts:', n_tot_pts,
                  'curr_rel:', curr_rel)
            pre_rel = curr_rel
        iter_ct += 1
 
    tot_vol = (max_sph_vol * curr_vol)
    print('tot_vol: %5.4E' % tot_vol)
#     for i in range(n_dims):
#         tot_vol *= diffs_arr[i]
#     return tot_vol
    return [shif_chull_pts, rand_pts_arr]
