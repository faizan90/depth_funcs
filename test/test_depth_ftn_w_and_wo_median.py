'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as guvecs,
    depth_ftn_mp as dftn,
    depth_ftn_mp_v2 as dftn2)

# raise Exception


def main():
    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 6
    n_vecs = int(1e4)
    n_cpus = 7

    rand_min = -3
    rand_max = +3
    n_rand_pts = int(1e4)

#     refr_pts = rand_min + ((rand_max - rand_min) *
#                            np.random.random((n_rand_pts, n_dims)))

    test_pts = rand_min + ((rand_max - rand_min) *
                           np.random.random((n_rand_pts, n_dims)))

    refr_pts = test_pts

    print('#### Unit vector generation test ####')
    usph_vecs = guvecs(n_vecs, n_dims, n_cpus)

    print('#### Depth test ####')
    _beg = timeit.default_timer()
    depths_1 = dftn(refr_pts, test_pts, usph_vecs, n_cpus, 1)
    _end = timeit.default_timer()
    print(f'Depths type 1 took {_end - _beg: 0.4f} secs!')

    print('#### Depth test ####')
    _beg = timeit.default_timer()
    depths_3 = dftn(refr_pts, test_pts, usph_vecs, n_cpus, 3)
    _end = timeit.default_timer()
    print(f'Depths type 3 took {_end - _beg: 0.4f} secs!')
#
    print('#### Depth test ####')
    _beg = timeit.default_timer()
    depths_2 = dftn2(refr_pts, test_pts, usph_vecs, n_cpus)
    _end = timeit.default_timer()
    print(f'Depths type 2 took {_end - _beg: 0.4f} secs!')

#     print('#### Depth test ####')
#     _beg = timeit.default_timer()
#     depths_1_sp = dftn(refr_pts, test_pts, usph_vecs, 1, 1)
#     _end = timeit.default_timer()
#     print(f'Depths type 1 took {_end - _beg: 0.4f} secs!')
#
#     print('#### Depth test ####')
#     _beg = timeit.default_timer()
#     depths_2_sp = dftn2(refr_pts, test_pts, usph_vecs, 1)
#     _end = timeit.default_timer()
#     print(f'Depths type 2 took {_end - _beg: 0.4f} secs!')

#     raise Exception

    idxs_11 = depths_1 == 1
    idxs_21 = depths_2 == 1
    idxs_31 = depths_3 == 1

    idxs_1x = depths_1 == 5
    idxs_2x = depths_2 == 5
    idxs_3x = depths_3 == 5

#     idxs_11_sp = depths_1_sp == 1
#     idxs_21_sp = depths_2_sp == 1

#     print(idxs_11.sum(), idxs_21.sum(), idxs_11_sp.sum(), idxs_21_sp.sum())

    depths_1[idxs_11]
    depths_2[idxs_11]
    depths_3[idxs_11]

    depths_1[idxs_21]
    depths_2[idxs_21]
    depths_3[idxs_21]

    depths_1[idxs_31]
    depths_2[idxs_31]
    depths_3[idxs_31]

    depths_1[idxs_1x]
    depths_2[idxs_1x]
    depths_3[idxs_1x]

    depths_1[idxs_2x]
    depths_2[idxs_2x]
    depths_3[idxs_2x]

    depths_1[idxs_3x]
    depths_2[idxs_3x]
    depths_3[idxs_3x]

#     for i in range(n_rand_pts):
#         print(depths_1[i], depths_3[i])

    assert np.all(depths_1 == depths_3)
    assert np.all(depths_1 == depths_2)
    assert np.all(depths_3 == depths_2)

    return


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('%s_log_%s.log' % (
                                    os.path.basename(__file__),
                                    datetime.now().strftime('%Y%m%d%H%M%S'))))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
