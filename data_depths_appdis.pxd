import numpy as np
cimport numpy as np

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef void cmpt_sorted_dot_prods_with_shrink(
        const DT_D[:, ::1] data,
              DT_D[:, ::1] sodp,
              DT_D[:, ::1] shdp,
        const DT_D[:, ::1] uvecs,
        const DT_UL n_pts,
              DT_UL n_cpus=?,
        )


cpdef np.ndarray get_sodp_depths(
        const DT_D[:, ::1] sodp,
        const DT_D[:, ::1] shdp,
        const DT_UL n_ref_pts,
        const DT_UL n_test_pts,
              DT_UL n_cpus=?,
        )
