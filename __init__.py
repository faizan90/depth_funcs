import pyximport
pyximport.install()

from .unit_sph_vecs import (gen_usph_vecs,
                            gen_usph_vecs_mp,
                            gen_usph_vecs_norm_dist,
                            gen_usph_vecs_norm_dist_mp)
from .data_depths import depth_ftn_mp
from .chull_volumes import cmpt_rand_pts_chull_vol

from .pyth_ftns import plot_depths_hist, depth_ftn_py
