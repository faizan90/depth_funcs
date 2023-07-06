'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from depth_funcs import (gen_usph_vecs_norm_dist,
                         cmpt_usph_chull_vol,
                         depth_ftn_mp)

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

    n_usph_vecs = int(1e5)
    n_dims = 5
    n_rand_pts = int(1e4)
    n_cpus = 7
    chk_iter = int(1e4)
    max_iters = 5

    os.chdir(main_dir)

    uvecs = gen_usph_vecs_norm_dist(n_usph_vecs, n_dims, n_cpus)
#     chull_pts = gen_usph_vecs_norm_dist(1000, n_dims, n_cpus)

    rnd_pts = np.random.random(size=(n_rand_pts, n_dims))

    chull_pts = rnd_pts[depth_ftn_mp(rnd_pts, rnd_pts, uvecs, n_cpus) == 1]
#
#     chull_pts = 1 + np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
#     chull_pts = np.array([[0.01, -0.99], [0.99, 0.01], [0.01, 0.99]], dtype=float)
#     chull_pts = 4 * np.array([[0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0]], dtype=float)
#     chull_pts = 2 * np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=float)
#     chull_pts = np.array([[-0.707, -0.707], [0.707, -0.707], [0.707, 0.707], [-0.707, 0.707]], dtype=float)
#     chull_pts = np.array([[0.707, -0.707], [0.707, 0.707], [-0.707, 0.707]], dtype=float)

#     chull_pts = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)

    shif_chull_pts, rnd_pts = cmpt_usph_chull_vol(chull_pts, uvecs, chk_iter, max_iters, n_cpus)

#     print(depth_ftn_mp(chull_pts, chull_pts, uvecs, n_cpus))

#     plt.scatter(uvecs[:, 0], uvecs[:, 1], alpha=0.5)
#     plt.scatter(rnd_pts[:, 0], rnd_pts[:, 1], alpha=0.01)
#     plt.scatter(chull_pts[:, 0], chull_pts[:, 1], alpha=0.1)
#     plt.scatter(shif_chull_pts[:, 0], shif_chull_pts[:, 1])
#
#     plt.grid()
#     plt.show()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
