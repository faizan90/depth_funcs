'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path
from itertools import product

import numpy as np
from scipy.spatial import ConvexHull

from depth_funcs import cmpt_rand_pts_chull_vol, gen_usph_vecs_mp, depth_ftn_mp

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

    n_cpus = 31
    n_unit_vecs = int(1e3)
    n_dims = 6
    chk_iter = int(1e3)
    max_iters = 100
    vol_tol = 0.001

    os.chdir(main_dir)

#     chull_arr = np.array(list(product(*[[1.0111, 1.0]] * n_dims)), dtype=float)
    chull_arr = np.array(list(product(*[[1.0, 0.0],
                                        [2.0, 3.0],
                                        [-4.0, -3.0],
                                        [-2.0, -5.0],
                                        [16.0, 19.0],
                                        [100.0, 102.0]])), dtype=float)
#     print('chull_arr:\n', chull_arr)
    print('chull shape before depth:', chull_arr.shape)
    chull_arr = chull_arr - np.random.random(size=chull_arr.shape) + np.random.random(size=chull_arr.shape)

    unit_vecs = gen_usph_vecs_mp(n_unit_vecs, n_dims, n_cpus)
    chull_arr = chull_arr[depth_ftn_mp(chull_arr, chull_arr, unit_vecs, n_cpus) == 1]
    print('chull shape after depths:', chull_arr.shape)
#     print(unit_vecs)
    print('unit_vecs shape:', unit_vecs.shape)
    unit_hull_vol = cmpt_rand_pts_chull_vol(chull_arr,
                                            unit_vecs,
                                            chk_iter,
                                            max_iters,
                                            n_cpus,
                                            vol_tol)

    print('unit_hull_vol:', unit_hull_vol)
    scipy_hull_vol = ConvexHull(chull_arr).volume
    print('scipy unit_hull_vol:', scipy_hull_vol)
    print('rand to scipy ratio:', unit_hull_vol / scipy_hull_vol)

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    if _save_log_:
        log_link.stop()
