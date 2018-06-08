'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from depth_funcs import depth_ftn_mp, gen_usph_vecs_norm_dist as gen_usph_vecs_mp

plt.ioff()

if __name__ == '__main__':
    # from datetime import datetime
    # from std_logger import StdFileLoggerCtrl

    # save all console activity to out_log_file
    # out_log_file = os.path.join(r'P:\\',
    #                             r'Synchronize',
    #                             r'python_script_logs',
    #                             ('xxx_log_%s.log' %
    #                              datetime.now().strftime('%Y%m%d%H%M%S')))
    # log_link = StdFileLoggerCtrl(out_log_file)
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    n_cpus = 7
    n_vecs = int(1e2)

    os.chdir(main_dir)

    unit_vecs = gen_usph_vecs_mp(n_vecs, 2, n_cpus)
    rand_pts = np.random.random(size=(100, 2))
    rand_pts_2 = np.random.random(size=(100, 2))
    chull_pts = rand_pts  # s[depth_ftn_mp(rand_pts, rand_pts, unit_vecs, n_cpus) == 1]

#     depths = depth_ftn_mp(chull_pts, chull_pts, unit_vecs, n_cpus)
#
#     corner_pts = chull_pts[depths == 1, :]
#
#     plt.scatter(chull_pts[:, 0], chull_pts[:, 1], label='chull')
#     plt.scatter(corner_pts[:, 0], corner_pts[:, 1], label='rands (d==1)', alpha=0.7)
    plt.scatter(unit_vecs[:, 0], unit_vecs[:, 1], alpha=0.7)

    plt.legend(framealpha=0.5)
    plt.grid()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    # log_link.stop()
    plt.show()
