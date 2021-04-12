'''
@author: Faizan-Uni-Stuttgart

Apr 9, 2021

7:45:48 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
    depth_ftn_mp)

DEBUG_FLAG = False


def obj_ftn(prms, ref_arr):

    sim_arr = np.empty_like(ref_arr)
    for i in range(sim_arr.shape[0]):
        sim_arr[i] = (prms * i).sum()

    obj_val = ((ref_arr - sim_arr) ** 2).sum()

    return obj_val


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 3
    n_vecs = int(1e4)
    n_cpus = 8

    n_rope_iters = 3
    n_vecs_per_iter = int(1e2)
    ref_arr = np.arange(100)

    cnvx_hull_cntn_tries = 10

    ref_min = -3
    ref_max = +3
    n_ref_pts = int(1e4)

    ref_pts = ref_min + (
        (ref_max - ref_min) * np.random.random((n_ref_pts, n_dims)))

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    obj_vals = np.empty(ref_pts.shape[0])
    for i in range(n_ref_pts):
        obj_vals[i] = obj_ftn(ref_pts[i,:], ref_arr)

    print('Initial obj vals min and max:', obj_vals.min(), obj_vals.max())

    best_pts = ref_pts[np.argsort(obj_vals)[:n_vecs_per_iter],:]

    for j in range(n_rope_iters):

        print('Iteration:', j)

        # Append the best points from the last iteration.
        new_pts = [best_pts]

        tot_n_new_points = best_pts.shape[0]
        for i in range(cnvx_hull_cntn_tries):
            print('Hull containment tries iteration:', i)

            adj_mins = best_pts.min(axis=0)
            adj_maxs = best_pts.max(axis=0)

            test_pts = adj_mins + (adj_maxs - adj_mins) * np.random.random(
                (n_ref_pts, adj_mins.shape[0]))

            depths = depth_ftn_mp(best_pts, test_pts, usph_vecs, n_cpus, 1)

            new_pts.append(test_pts[depths > 0])

            tot_n_new_points += (depths > 0).sum()

            if tot_n_new_points >= n_ref_pts:
                print(f'Got a total of {tot_n_new_points} inside the hull.')
                break

            else:
                print(
                    f'Got {(depths > 0).sum()} points inside the hull in '
                    f'this iteration.')

        if tot_n_new_points < n_ref_pts:
            print('Could not sample enough points inside the hull!')

            break

        ref_pts = np.concatenate(new_pts, axis=0)

        obj_vals = np.empty(ref_pts.shape[0])
        for i in range(n_ref_pts):
            obj_vals[i] = obj_ftn(ref_pts[i,:], ref_arr)

        print(
            'Obj vals min and max for this iteration:',
            obj_vals.min(),
            obj_vals.max())

        best_pts = ref_pts[np.argsort(obj_vals)[:n_vecs_per_iter],:]

        print('\n\n')

    obj_vals = np.empty(best_pts.shape[0])
    for i in range(obj_vals.shape[0]):
        obj_vals[i] = obj_ftn(best_pts[i,:], ref_arr)

    print('Final obj vals min and max:', obj_vals.min(), obj_vals.max())

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
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
