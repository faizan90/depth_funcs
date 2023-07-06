# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)
# cython: embedsignature=True

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef DT_D cmpt_rand_pts_chull_vol(
    const DT_D[:, ::1] in_chull_pts,
    const DT_D[:, ::1] uvecs,
          DT_UL chk_iter,
          DT_UL max_iters, 
          DT_UL n_cpus=?,
          DT_D vol_tol=?)
