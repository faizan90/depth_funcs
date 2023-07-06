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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from depth_funcs import (
    # plot_depths_hist,
    # depth_ftn_py,
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
    depth_ftn_mp)

np.set_printoptions(
    precision=3,
    # threshold=2000,
    # linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})

DEBUG_FLAG = True


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\depths\sample_in_chull')
    os.chdir(main_dir)

    path_to_chull_pts = Path(r'chull_pts.txt')

    n_vecs = int(1e6)
    n_cpus = 16

    n_new_pts_fin = int(1e4)

    out_fig_path = Path(rf'hbv_2d_scatter_v9_14.png')
    #==========================================================================

    rand_pts = np.loadtxt(
        path_to_chull_pts, delimiter='\t')  # , usecols=np.arange(5, 11))

    print(rand_pts.shape)

    n_dims = rand_pts.shape[1]

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    print('Sampling on chull...')

    beg_timer = timeit.default_timer()
    pts_chull = sample_on_chull_v5(rand_pts, usph_vecs, n_new_pts_fin, n_cpus)
    end_timer = timeit.default_timer()

    print(f'Sampling on chull took: {end_timer - beg_timer:0.2f} secs.')

    print('Sampling in chull...')

    beg_timer = timeit.default_timer()
    pts_deep = sample_in_chull(
        pts_chull, usph_vecs, n_new_pts_fin, n_cpus, True)
    end_timer = timeit.default_timer()

    print(f'Sampling in chull took: {end_timer - beg_timer:0.2f} secs.')

    plot_prms_scatters(pts_chull, pts_deep, out_fig_path)
    return


def sample_in_chull(ref_pts_chull, uvecs, n_pts_deep, n_cpus, on_chull_flag):

    n_dims = ref_pts_chull.shape[1]

    n_chull_pts_avail = ref_pts_chull.shape[0]

    deep_pts = np.empty((n_pts_deep, n_dims))

    tst_pts = np.empty_like(ref_pts_chull)

    end_idx = 0
    while True:

        for i in range(0, tst_pts.shape[0]):

            j = np.random.randint(0, n_chull_pts_avail)

            k = np.random.randint(0, n_chull_pts_avail)
            while (k == j):
                k = np.random.randint(0, n_chull_pts_avail)

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        if on_chull_flag:
            deep_idxs = np.arange(depths.size)

        else:
            deep_idxs = np.where(depths > 1)[0]

        beg_idx = end_idx

        end_idx += deep_idxs.size

        if end_idx > n_pts_deep:

            deep_idxs = deep_idxs[end_idx - n_pts_deep:]

            end_idx = n_pts_deep

        deep_pts[beg_idx:end_idx,:] = tst_pts[deep_idxs,:]

        print(end_idx)

        if end_idx == n_pts_deep:
            break

    return deep_pts


def sample_on_chull(ref_pts, uvecs, n_pts_chull, n_cpus):

    n_dims = ref_pts.shape[1]

    chull_pts = np.empty((n_pts_chull, n_dims))

    tst_pts = ref_pts.copy()

    ref_pts_chull = ref_pts.copy()

    end_idx = 0
    while True:
        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        chull_idxs = np.where(depths == 1)[0]

        if end_idx == 0:
            ref_pts_chull = ref_pts[chull_idxs,:].copy()

            n_chull_pts_avail = ref_pts_chull.shape[0]
            assert n_chull_pts_avail > 1

        beg_idx = end_idx

        end_idx += chull_idxs.size

        if end_idx > n_pts_chull:

            chull_idxs = chull_idxs[end_idx - n_pts_chull:]

            end_idx = n_pts_chull

        chull_pts[beg_idx:end_idx,:] = tst_pts[chull_idxs,:]

        print(end_idx)

        if end_idx == n_pts_chull:
            break

        for i in range(0, tst_pts.shape[0]):

            j = np.random.randint(0, n_chull_pts_avail)

            k = np.random.randint(0, n_chull_pts_avail)
            while (k == j):
                k = np.random.randint(0, n_chull_pts_avail)

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

    return chull_pts


def sample_on_chull_v2(ref_pts, uvecs, n_pts_chull, n_cpus):

    n_dims = ref_pts.shape[1]

    chull_pts = np.empty((n_pts_chull, n_dims))

    tst_pts = ref_pts.copy()

    ref_pts_chull = ref_pts.copy()

    end_idx = 0
    while True:
        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        chull_idxs = np.where(depths == 1)[0]

        if end_idx == 0:
            ref_pts_chull = ref_pts[chull_idxs,:].copy()

            n_chull_pts_avail = ref_pts_chull.shape[0]
            assert n_chull_pts_avail > 1

        beg_idx = end_idx

        end_idx += chull_idxs.size

        if end_idx > n_pts_chull:

            chull_idxs = chull_idxs[end_idx - n_pts_chull:]

            end_idx = n_pts_chull

        chull_pts[beg_idx:end_idx,:] = tst_pts[chull_idxs,:]

        print(end_idx)

        if end_idx == n_pts_chull:
            break

        for i in range(0, tst_pts.shape[0]):

            j = np.random.randint(0, n_chull_pts_avail)

            dim_rand = np.random.randint(0, n_dims)

            dists_d = ref_pts_chull[j, dim_rand] - ref_pts_chull[:, dim_rand]

            rand_sign = np.random.choice([-1, +1])

            dists_d *= rand_sign

            try:
                k = np.argmin(dists_d[dists_d > 0])

                while k == j:
                    dists_d[k] = +np.inf

                    k = np.argmin(dists_d[dists_d > 0])

            except ValueError:

                k = np.argmax(dists_d[dists_d < 0])

                while k == j:
                    dists_d[k] = -np.inf

                    k = np.argmax(dists_d[dists_d < 0])

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

    return chull_pts


def sample_on_chull_v3(
        ref_pts, uvecs, n_pts_chull, n_cpus, ovr_smplg_ratio=0.0):

    n_dims = ref_pts.shape[1]

    chull_pts = np.empty((n_pts_chull, n_dims))

    ref_pts_chull = ref_pts

    tst_pts = ref_pts_chull

    end_idx = 0

    loop_ctr = 0
    while True:

        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        chull_idxs = np.where(depths == 1)[0]

        if loop_ctr == 0:
            ref_pts_chull = ref_pts_chull[chull_idxs,:].copy()

            n_chull_pts_avail = ref_pts_chull.shape[0]
            assert n_chull_pts_avail > 1

        print('chull_idxs.size:', chull_idxs.size)

        beg_idx = end_idx

        end_idx += chull_idxs.size

        if end_idx > n_pts_chull:

            chull_idxs = chull_idxs[end_idx - n_pts_chull:]

            end_idx = n_pts_chull

        chull_pts[beg_idx:end_idx,:] = tst_pts[chull_idxs,:]

        print('end_idx:', end_idx)

        if end_idx == n_pts_chull:
            break

        if loop_ctr == 0:
            tst_pts = np.empty(
                (int(n_pts_chull * (1 + ovr_smplg_ratio)), n_dims))

        for i in range(0, tst_pts.shape[0]):

            j = np.random.randint(0, n_chull_pts_avail)

            dim_rand = np.random.randint(0, n_dims)

            dists_d = ref_pts_chull[j,:] - ref_pts_chull

            dists_d = np.delete(dists_d, dim_rand, axis=1)

            dists_d[j,:] = np.inf

            dists_d = ((dists_d ** 2).sum(axis=1)) ** 0.5

            dists_j = np.abs(
                ref_pts_chull[j, dim_rand] - ref_pts_chull[:, dim_rand])

            dists_j[j] = np.inf

            k = np.argmin(((dists_d ** 2) + (dists_j ** 2)) ** 0.5)

            assert k != j

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

        loop_ctr += 1

    return chull_pts


def sample_on_chull_v4(
        ref_pts, uvecs, n_pts_chull, n_cpus, ovr_smplg_ratio=0.0):

    n_dims = ref_pts.shape[1]

    chull_pts = np.empty((n_pts_chull, n_dims))

    ref_pts_chull = ref_pts

    tst_pts = ref_pts_chull

    end_idx = 0

    loop_ctr = 0
    while True:

        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        chull_idxs = np.where(depths == 1)[0]

        if loop_ctr == 0:
            ref_pts_chull = ref_pts_chull[chull_idxs,:].copy()

            n_chull_pts_avail = ref_pts_chull.shape[0]
            assert n_chull_pts_avail > 1

        print('chull_idxs.size:', chull_idxs.size)

        beg_idx = end_idx

        end_idx += chull_idxs.size

        if end_idx > n_pts_chull:

            chull_idxs = chull_idxs[end_idx - n_pts_chull:]

            end_idx = n_pts_chull

        chull_pts[beg_idx:end_idx,:] = tst_pts[chull_idxs,:]

        print('end_idx:', end_idx)

        if end_idx == n_pts_chull:
            break

        if loop_ctr == 0:
            tst_pts = np.empty(
                (int(n_pts_chull * (1 + ovr_smplg_ratio)), n_dims))

        vstd_pts = np.zeros(tst_pts, dtype=bool)

        for i in range(0, tst_pts.shape[0]):

            j = np.random.randint(0, n_chull_pts_avail)

            assert np.any(~vstd_pts)

            while vstd_pts[j]:
                j = np.random.randint(0, n_chull_pts_avail)

            dists_d = ((
                (ref_pts_chull[j,:] - ref_pts_chull) ** 2).sum(axis=1)) ** 0.5

            dists_d[j] = np.inf

            k = np.argmin(dists_d)

            assert k != j

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

        loop_ctr += 1

    return chull_pts


def sample_on_chull_v5(ref_pts, uvecs, n_pts_chull, n_cpus):

    n_dims = ref_pts.shape[1]

    chull_pts = np.empty((n_pts_chull, n_dims))

    ref_pts_chull = ref_pts

    tst_pts = ref_pts_chull

    n_nrst_pts = n_dims * 5

    end_idx = 0

    loop_ctr = 0
    while True:

        depths = depth_ftn_mp(ref_pts_chull, tst_pts, uvecs, n_cpus, 1)

        chull_idxs = np.where(depths == 1)[0]

        if loop_ctr == 0:
            ref_pts_chull = ref_pts_chull[chull_idxs,:].copy()

            n_chull_pts_avail = ref_pts_chull.shape[0]
            assert n_chull_pts_avail > 1

        print('chull_idxs.size:', chull_idxs.size)

        beg_idx = end_idx

        end_idx += chull_idxs.size

        if end_idx > n_pts_chull:

            chull_idxs = chull_idxs[end_idx - n_pts_chull:]

            end_idx = n_pts_chull

        chull_pts[beg_idx:end_idx,:] = tst_pts[chull_idxs,:]

        print('end_idx:', end_idx)

        if end_idx == n_pts_chull:
            break

        vstd_pts = np.zeros((n_chull_pts_avail, n_nrst_pts), dtype=bool)

        tst_pts = np.empty((n_pts_chull, n_dims))

        for i in range(tst_pts.shape[0]):

            if np.all(vstd_pts):

                tst_pts = tst_pts[:i,:].copy()

                print(f'Need to resample ({i})...')

                break

            j = np.random.randint(n_chull_pts_avail)

            d = np.random.randint(n_nrst_pts)

            while vstd_pts[j, d]:

                j = np.random.randint(n_chull_pts_avail)

                d = np.random.randint(n_nrst_pts)

            dists_d = (ref_pts_chull[j,:] - ref_pts_chull)

            dists_d = ((dists_d ** 2).sum(axis=1)) ** 0.5

            dists_d[j] = np.inf

            k = np.argsort(dists_d)[d]

            assert k != j

            vstd_pts[j, d] = True

            rand_t = np.random.random()
            for m in range(ref_pts_chull.shape[1]):
                tst_pts[i, m] = (
                    (ref_pts_chull[j, m] * rand_t) +
                    (ref_pts_chull[k, m] * (1 - rand_t)))

        loop_ctr += 1

    return chull_pts


def plot_prms_scatters(chull_pts, test_pts, out_fig_path):

    chull_min = +0
    chull_max = +1
    n_dims = test_pts.shape[-1]

    plt.figure(figsize=(8.5, 8.5), dpi=200)

    grid_axes = GridSpec(n_dims, n_dims)

    for i in range(n_dims):
        for j in range(n_dims):
            if i >= j:
                continue

            ax = plt.subplot(grid_axes[i, j])

            ax.set_aspect('equal', 'datalim')

            ax.set_xlim(chull_min, chull_max)
            ax.set_ylim(chull_min, chull_max)

            ax.scatter(
                test_pts[:, i],
                test_pts[:, j],
                s=3,
                color='k',
                alpha=0.03,
                edgecolors='none')

            ax.text(
                0.95,
                0.95,
                f'({i}, {j})',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_xticklabels([])

            ax.set_yticks([])
            ax.set_yticklabels([])

    if chull_pts is not None:
        for i in range(n_dims):
            for j in range(n_dims):
                if i >= j:
                    continue

                ax = plt.subplot(grid_axes[j, i])

                ax.set_aspect('equal', 'datalim')

                ax.set_xlim(chull_min, chull_max)
                ax.set_ylim(chull_min, chull_max)

                ax.scatter(
                    chull_pts[:, i],
                    chull_pts[:, j],
                    s=3,
                    color='r',
                    alpha=0.03,
                    edgecolors='none')

                # ax.scatter(
                #     test_pts[:, i],
                #     test_pts[:, j],
                #     s=5,
                #     color='k',
                #     alpha=0.05,
                #     zorder=2,
                #     edgecolors='none')

                ax.text(
                    0.95,
                    0.95,
                    f'({i}, {j})',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_xticklabels([])

                ax.set_yticks([])
                ax.set_yticklabels([])

    # plt.show()

    plt.savefig(out_fig_path, bbox_inches='tight')

    plt.clf()

    plt.close('all')
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
