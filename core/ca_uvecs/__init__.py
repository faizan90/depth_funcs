# -*- coding: utf-8 -*-

'''
Created on Feb 25, 2023

@author: Faizan_TR
'''

import pyximport
pyximport.install()

from .unit_sph_vecs import (
    gen_usph_vecs,
    gen_usph_vecs_mp,
    gen_usph_vecs_norm_dist,
    gen_usph_vecs_norm_dist_mp)
