import pyximport
pyximport.install()

from .depth_ftns_main import (gen_usph_vecs,
                              gen_usph_vecs_mp,
                              depth_ftn_mp,
                              cmpt_rand_pts_chull_vol)

from .pyth_ftns import plot_depths_hist, depth_ftn_py
