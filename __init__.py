import pyximport
pyximport.install()

from .depth_cy_ftns import gen_usph_vecs, gen_usph_vecs_mp, depth_ftn_mp
