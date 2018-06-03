import numpy as np
cimport numpy as np

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef np.ndarray depth_ftn_mp(
    const DT_D[:, :] ref, 
    const DT_D[:, :] test, 
    const DT_D[:, :] uvecs,
          DT_UL n_cpus=?)
