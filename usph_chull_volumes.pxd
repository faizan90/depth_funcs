from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef cmpt_usph_chull_vol(
    const DT_D[:, ::1] in_chull_pts,
    const DT_D[:, ::1] uvecs,
          DT_UL chk_iter,
          DT_UL max_iters, 
          DT_UL n_cpus=?,
          DT_D vol_tol=?)
