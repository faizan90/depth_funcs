'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path
from itertools import permutations as perms

import matplotlib.pyplot as plt

import numpy as np

import pyximport
pyximport.install()

from depth_funcs import depth_ftn_mp


def depth_ftn_py(x, y, ei):
    mins = x.shape[0] * np.ones((y.shape[0],))  # initial value

    for i in ei:  # iterate over unit vectors
        d = np.dot(i, x.T)  # scalar product

        dy = np.dot(i, y.T)  # scalar product

        # d = d[np.argsort(d)]  # argsort gives the sorting indices then we used it to sort d
        d.sort()

        dy_med = np.median(dy)
        dy = ((dy - dy_med) * (1 - (1e-7))) + dy_med

        numl = np.searchsorted(d, dy)  # find the index of each project y in x to preserve order
        numg = d.shape[0] - numl  # numl is number of points less then projected y

        mins = np.min(np.vstack([mins, np.min(np.vstack([numl, numg]), axis=0)]), 0)  # find new min

    return mins.astype(np.uint64)


def plot_depths_hist(x_arr_left,
                     x_arr_right,
                     y_arr_left,
                     y_arr_right,
                     eis_arr,
                     title_lab,
                     out_fig_loc,
                     n_cpus,
                     fig_size,
                     labs=None):

    depth_arrs_list = []
    hist_labs = []

    out_fig_loc = str(out_fig_loc)

    hist_bins = np.concatenate((np.arange(5), [10000]))

    thresh_depth = hist_bins[-3]
    n_dims = x_arr_left.shape[1]

    arr_perms = list(perms([x_arr_left,
                            x_arr_right,
                            y_arr_left,
                            y_arr_right], 2))

    if labs is None:
        labs = ['x_left', 'x_right', 'y_left', 'y_right']
        labs_perms = list(perms(labs, 2))
    else:
        assert len(labs) == 4
        labs_perms = list(perms(labs, 2))

    for arrs, _labs in zip(arr_perms, labs_perms):
        if arrs[0].shape[0] and arrs[1].shape[0]:
            _arr = depth_ftn_mp(arrs[1], arrs[0], eis_arr, n_cpus)
            _arr[_arr > thresh_depth] = thresh_depth
            depth_arrs_list.append(_arr)
            hist_labs.append('%s_in_%s' % _labs)

    _out_path, _out_ext = out_fig_loc.rsplit('.', 1)

    plt.figure(figsize=fig_size)
    rwidth = 0.98

    items_per_fig = 4

    for i, (mins, hist_lab) in enumerate(zip(depth_arrs_list, hist_labs), 1):
        plt.hist(mins,
                 bins=hist_bins,
                 alpha=0.3,
                 density=True,
                 rwidth=rwidth,
                 label=hist_lab,
                 align='mid')
        rwidth -= 0.17

        if (not (i % items_per_fig)) or (i == len(hist_labs)):
            hist_title = ''
            hist_title += (('%s\n'
                            '%d %d-dimensional unit vectors used\n'
                            'n_%s=%d, n_%s=%d, '
                            'n_%s=%d, n_%s=%d') %
                           (title_lab,
                            eis_arr.shape[0],
                            n_dims,
                            labs[0],
                            x_arr_left.shape[0],
                            labs[1],
                            x_arr_right.shape[0],
                            labs[2],
                            y_arr_left.shape[0],
                            labs[3],
                            y_arr_right.shape[0]))
            plt.title(hist_title)

            plt.xticks(hist_bins[:-2] + 0.5,
                       (hist_bins[:-3].tolist() +
                        ['>=%d' % thresh_depth]))
            plt.xlim(hist_bins[0], hist_bins[-2])
            plt.ylim(0, 1)

            plt.legend()
            plt.grid()
            plt.savefig(_out_path + ('_%0.2d.%s' % (i, _out_ext)), bbox_inches='tight')
            rwidth = 0.98
            plt.clf()

    plt.close()
    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
