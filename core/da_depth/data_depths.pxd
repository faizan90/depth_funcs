# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types(None)
# cython: embedsignature=True

import numpy as np
cimport numpy as np

from ..aa_basic.dtypes cimport DT_D, DT_UL, DT_ULL


cpdef np.ndarray depth_ftn_mp(
    const DT_D[:, ::1] ref, 
    const DT_D[:, ::1] test, 
    const DT_D[:, ::1] uvecs,
          long n_cpus=?,
          DT_UL depth_type=?) except +


cpdef np.ndarray depth_ftn_mp_v2(
    const DT_D[:, ::1] ref, 
    const DT_D[:, ::1] test, 
    const DT_D[:, ::1] uvecs,
          DT_UL n_cpus=?)

cpdef void sort_arr(DT_D[::1] in_arr)
