cpdef DT_D cmpt_rand_pts_chull_vol(
    const DT_D[:, :] in_chull_pts,
    const DT_D[:, :] unit_vecs,
          DT_UL chk_iter,
          DT_UL max_iters, 
          DT_UL n_cpus=1,
          DT_D vol_tol=0.01)
