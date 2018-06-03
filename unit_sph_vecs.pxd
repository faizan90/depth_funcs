import numpy as np
cimport numpy as np

from .dtypes cimport DT_D, DT_UL, DT_ULL


cpdef np.ndarray gen_usph_vecs(n_vecs, n_dims)

cpdef np.ndarray gen_usph_vecs_mp(DT_UL n_vecs, DT_UL n_dims, DT_UL n_cpus)
