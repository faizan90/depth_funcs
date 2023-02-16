# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Feb 14, 2023

10:47:14 AM

Keywords:

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
# import matplotlib.pyplot as plt

np.set_printoptions(
    precision=3,
    # threshold=2000,
    # linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})

from depth_funcs import (
    # plot_depths_hist,
    # depth_ftn_py,
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
    depth_ftn_mp)

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 8
    n_vecs = int(5e5)
    n_cpus = 8

    n_rand_pts = int(4e3)
    #==========================================================================

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    rand_pts = gen_usph_vecs_mp(n_rand_pts, n_dims, n_cpus)

    # depths = depth_ftn_mp(rand_pts, rand_pts, usph_vecs, n_cpus, 1)
    #
    # depths_unq, depths_cts = np.unique(depths, return_counts=True)
    #
    # for depth_unq, depth_ct in zip(depths_unq, depths_cts):
    #     print(depth_unq, depth_ct)

    new_pts = np.empty_like(rand_pts)

    for i in range(rand_pts.shape[0]):

        j = np.random.randint(0, rand_pts.shape[0])
        while j == i:
            j = np.random.randint(0, rand_pts.shape[0])

        k = np.random.randint(0, rand_pts.shape[0])
        while (k == i) or (k == j):
            k = np.random.randint(0, rand_pts.shape[0])

        rand_t = 0.1 + ((0.8) * np.random.random())
        for m in range(rand_pts.shape[1]):
            new_pts[i, m] = (rand_pts[j, m] * rand_t) + (rand_pts[k, m] * (1 - rand_t))

    depths = depth_ftn_mp(rand_pts, new_pts, usph_vecs, n_cpus, 1)

    depths_unq, depths_cts = np.unique(depths, return_counts=True)

    for depth_unq, depth_ct in zip(depths_unq[:5], depths_cts[:5]):
        print(depth_unq, depth_ct)

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
