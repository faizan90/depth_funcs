'''
@author: Faizan-Uni-Stuttgart

Jul 7, 2020

12:42:31 PM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp,
    depth_ftn_mp)

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_vecs = int(1e7)
    n_cpus = 7
    n_test_pts = 300

    ucube_pts = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        ], dtype=np.float64)

    n_dims = ucube_pts.shape[1]

#     rand_pts = -0.1 + np.random.random(size=(n_test_pts, n_dims)) * 1.2
#
#     corr = 0.5
#     rand_pts[:, 1] = (corr * rand_pts[:, 0]) + (((1 - corr ** 2) ** 0.5) * rand_pts[:, 1])

    f1 = r"P:\Synchronize\IWS\Projects\2016_DFG_SPATE\data\bjoern\joint_var_events\joint_416.txt"
    df1 = pd.read_csv(f1, sep=';', index_col=0)[['Vp2', 'Pmax2', 'sm_all', 'melt2']]

    rand_pts = df1.values.copy(order='c')

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    depths = depth_ftn_mp(rand_pts, rand_pts, usph_vecs, n_cpus, 1)

#     print(depths)

#     for i in range(n_test_pts):
#         print(rand_pts[i], depths[i])

    for j in range(n_dims):
        print(
            j,
            depths[np.argmin(rand_pts[:, j])],
            depths[np.argmax(rand_pts[:, j])])

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

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
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
