'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as get_uvecs,
    cmpt_sorted_dot_prods_with_shrink, get_sodp_depths,
    depth_ftn_mp_v2 as get_depths)


def main():
    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_dims = 5
    n_vecs = int(5e4)
    n_cpus = 7

    rand_min = -3
    rand_max = +3
    n_rand_pts = int(1e4)

    hide_pts = 10

    os.chdir(main_dir)

    assert hide_pts < n_rand_pts
    rand_pts = rand_min + ((rand_max - rand_min) *
                           np.random.random((n_rand_pts, n_dims)))

    print('#### Unit vector generation test ####')

    _beg = timeit.default_timer()
    usph_vecs = get_uvecs(n_vecs, n_dims, n_cpus)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    mags = np.sqrt((usph_vecs ** 2).sum(axis=1))
    idxs = (mags > 1.000001)
    print('%d out of %d unit vectors have lengths greater than 1!' %
          (idxs.sum(), int(n_vecs)))

    print('\n')
    print('#### Sorted dot product gen test ####')
    sdp = np.empty((n_vecs, n_rand_pts), dtype=np.float64, order='c')
    shdp = sdp.copy()

    _beg = timeit.default_timer()
    cmpt_sorted_dot_prods_with_shrink(
        rand_pts, sdp, shdp, usph_vecs, n_rand_pts - hide_pts, n_cpus)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    print('\n')
    print('#### Sorted dot product depth test ####')

    _beg = timeit.default_timer()
    sdp_depths = get_sodp_depths(
        sdp, shdp, n_rand_pts - hide_pts, n_rand_pts - hide_pts, n_cpus)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    print('\n')
    print('#### Traditonal depth test ####')
    _beg = timeit.default_timer()
    tdl_depths = get_depths(
        rand_pts[:-hide_pts], rand_pts[:-hide_pts], usph_vecs, n_cpus)
    _end = timeit.default_timer()
    print(f'Took {_end - _beg: 0.4f} secs!')

    assert np.all(sdp_depths == tdl_depths)

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
