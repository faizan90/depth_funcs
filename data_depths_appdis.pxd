import numpy as np
cimport numpy as np

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef void cmpt_sorted_dot_prods_with_shrink(
        const DT_D[:, ::1] data,
              DT_D[:, ::1] sdp,
              DT_D[:, ::1] shdp,
        const DT_D[:, ::1] uvecs,
              DT_UL n_cpus=?,
        )


cpdef np.ndarray get_sdp_depths(
        const DT_D[:, ::1] sdp,
        const DT_D[:, ::1] shdp,
              DT_UL n_cpus=?,
        )
