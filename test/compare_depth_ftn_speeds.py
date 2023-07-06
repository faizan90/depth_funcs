'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np

from depth_funcs import gen_usph_vecs_norm_dist, depth_ftn_mp, depth_ftn_mp_v2

if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('xxx_log_%s.log' %
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        log_link = StdFileLoggerCtrl(out_log_file)
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    n_vecs = int(1e4)
    n_dims = 6
    n_rand_pts = int(1e5)
    n_cpus = 7

    os.chdir(main_dir)

    uvecs = gen_usph_vecs_norm_dist(n_vecs, n_dims, n_cpus)

    rand_pts_1 = -2 + (4 * np.random.random(size=(n_rand_pts, n_dims)))
    rand_pts_2 = -2 + (4 * np.random.random(size=(n_rand_pts, n_dims)))

    _strt = timeit.default_timer()
    x = depth_ftn_mp(rand_pts_1, rand_pts_2, uvecs, n_cpus)
    _end = timeit.default_timer()
    print('orignal: %0.5f' % (_end - _strt))
    print(np.histogram(x))

    _strt = timeit.default_timer()
    x = depth_ftn_mp_v2(rand_pts_1, rand_pts_2, uvecs, n_cpus)
    _end = timeit.default_timer()
    print('new: %0.5f' % (_end - _strt))
    print(np.histogram(x))

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    if _save_log_:
        log_link.stop()
