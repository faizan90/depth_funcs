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
import matplotlib.pyplot as plt

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
    depth_ftn_mp)

plt.ioff()

DEBUG_FLAG = False


def model(i, prms):

    return (prms * i).sum()


def obj_ftn(prms, ref_arr):

    sim_arr = np.empty_like(ref_arr)
    for i in range(sim_arr.shape[0]):
        sim_arr[i] = model(i, prms)

    obj_val = ((ref_arr - sim_arr) ** 2).sum()

    return obj_val


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 3
    n_uvecs = int(1e3)
    n_cpus = 8

    n_rope_iters = 5
    n_best_pts_per_iter = int(1e2)
    ref_arr = np.arange(50)

    cnvx_hull_cntn_tries = 5
    n_test_pts_per_try = int(1e4)

    ref_min = -3
    ref_max = +3
    n_ref_pts = int(1e3)

    ref_pts = ref_min + (
        (ref_max - ref_min) * np.random.random((n_ref_pts, n_dims)))

    usph_vecs = gen_usph_vecs_mp(n_uvecs, n_dims, n_cpus)

    obj_vals = np.empty(ref_pts.shape[0])
    for i in range(obj_vals.shape[0]):
        obj_vals[i] = obj_ftn(ref_pts[i,:], ref_arr)

    print('Initial obj vals min and max:', obj_vals.min(), obj_vals.max())

    best_pts = ref_pts[np.argsort(obj_vals)[:n_best_pts_per_iter],:]

    obj_min_global = obj_vals.min()
    best_global_pt = best_pts[0,:].copy()

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
                (n_test_pts_per_try, adj_mins.shape[0]))

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
        for i in range(obj_vals.shape[0]):
            obj_vals[i] = obj_ftn(ref_pts[i,:], ref_arr)

        print(
            'Obj vals min and max for this iteration:',
            obj_vals.min(),
            obj_vals.max())

        best_pts = ref_pts[np.argsort(obj_vals)[:n_best_pts_per_iter],:]

        if obj_vals.min() < obj_min_global:
            obj_min_global = obj_vals.min()
            best_global_pt = best_pts[0,:].copy()

        elif obj_vals.min() == obj_min_global:
            pass

        else:
            best_pts[-1,:] = best_global_pt

        print('\n\n')

    obj_vals = np.empty(best_pts.shape[0])
    for i in range(obj_vals.shape[0]):
        obj_vals[i] = obj_ftn(best_pts[i,:], ref_arr)

    print('Final obj vals min and max:', obj_vals.min(), obj_vals.max())

    print('Best prms:')
    print(best_pts)

    plt.figure(figsize=(10, 10))

    x_arr = np.arange(ref_arr.shape[0])

    for i in range(obj_vals.shape[0]):

        sim_arr = np.array([
            model(best_pts[i,:], j) for j in range(ref_arr.shape[0])])

        plt.scatter(x_arr, sim_arr, alpha=0.2, c='k')

    plt.plot(ref_arr, alpha=0.95, c='r', label='ref')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.show()
    plt.close()
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
