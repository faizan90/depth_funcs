'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:+0.3f}'.format})

# compile_cuda_flag = False
# # compile_cuda_flag = True
#
# if compile_cuda_flag:
#     from faizpy import create_cuda_lib_files
#     create_cuda_lib_files(['data_depths_cftns'])

# raise Exception
# import pyximport
# pyximport.install()

from depth_funcs import (plot_depths_hist,
                         depth_ftn_py,
                         gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
                         depth_ftn_mp)

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    n_dims = 3
    n_vecs = int(1e5)
    n_cpus = 7

    rand_min = -3
    rand_max = +3
    n_rand_pts = int(1e4)

    plot_depths_hist_flag = False

    os.chdir(main_dir)

    rand_pts = rand_min + ((rand_max - rand_min) *
                           np.random.random((n_rand_pts, n_dims)))

    rand_pts[0] = rand_min
    rand_pts[1] = rand_max

    test_pts = np.array([[0.5 * (rand_max + rand_min)] * n_dims,
                         rand_pts[0],
                         rand_pts[1],
                         [rand_min - 1] * n_dims,
                         [rand_max + 1] * n_dims], dtype=float)

    test_pts_msgs = ['should be close to %d' % int(0.5 * n_rand_pts),
                     'should be 1',
                     'should be 1',
                     'should be 0',
                     'should be 0']

    assert test_pts.shape[0] == len(test_pts_msgs)

    print('#### Unit vector generation test ####')
    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    mags = np.sqrt((usph_vecs ** 2).sum(axis=1))
    idxs = (mags > 1.000001)
    print('%d out of %d unit vectors have lengths greater than 1!' %
          (idxs.sum(), int(n_vecs)))

#     plt.scatter(usph_vecs[:, 0], usph_vecs[:, 1], alpha=0.01)
#     plt.show(block=True)
#     raise Exception

    print('\nHistograms of each dimension...')
    for i in range(n_dims):
        print('Dimension no:', i + 1)
        hists = np.histogram(usph_vecs[:, i], bins=np.linspace(-1.0, 1.0, 21))
        print(hists[1])
        print(hists[0])
        print((hists[0] - (n_vecs / hists[0].shape[0])) / (n_vecs / hists[0].shape[0]))
        print('\n')

    print('#### Depth test ####')
    _beg = timeit.default_timer()
    depth_cy1 = depth_ftn_mp(rand_pts, test_pts, usph_vecs, n_cpus, 1)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    for i in range(test_pts.shape[0]):
        print('Depth of point %s in %d random points: %d (%s)' %
              (str(test_pts[i]), n_rand_pts, depth_cy1[i], test_pts_msgs[i]))

    print('#### Depth test ####')
    _beg = timeit.default_timer()
    depth_cy3 = depth_ftn_mp(rand_pts, test_pts, usph_vecs, n_cpus, 3)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    for i in range(test_pts.shape[0]):
        print('Depth of point %s in %d random points: %d (%s)' %
              (str(test_pts[i]), n_rand_pts, depth_cy3[i], test_pts_msgs[i]))

    time.sleep(1)
    raise Exception
#     _idxs = (rand_pts[:, 0] > (rand_min + 1))
#     depth_cy = depth_ftn_mp(rand_pts, rand_pts[_idxs, :], usph_vecs, n_cpus)
#     print('Points with depths of 0:', depth_cy[depth_cy == 0].sum())

    if plot_depths_hist_flag:
        print('\n')
        print('#### Plotting depths histogram ####')
        rand_test_pts = rand_min + ((rand_max - rand_min) *
                                    np.random.random((n_rand_pts, n_dims)))

        x_mid_idx = int(0.5 * n_rand_pts)
        y_mid_idx = int(0.5 * rand_test_pts.shape[0])
        hi_val_thresh = 0
        out_fig_loc = main_dir / 'test_fig.png'
        fig_size = (13, 7)
        title_lab = 'Random test case'

        plot_depths_hist(rand_pts[:x_mid_idx],
                         rand_pts[x_mid_idx:],
                         rand_test_pts[:y_mid_idx],
                         rand_test_pts[y_mid_idx:],
                         usph_vecs,
                         title_lab,
                         out_fig_loc,
                         n_cpus,
                         fig_size)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
