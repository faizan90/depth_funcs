import numpy as np
cimport numpy as np

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef np.ndarray depth_ftn_mp(
    const DT_D[:, ::1] ref, 
    const DT_D[:, ::1] test, 
    const DT_D[:, ::1] uvecs,
          long n_cpus=?) except +


cpdef np.ndarray depth_ftn_mp_v2(
    const DT_D[:, :] ref, 
    const DT_D[:, :] test, 
    const DT_D[:, :] uvecs,
          DT_UL n_cpus=?)
