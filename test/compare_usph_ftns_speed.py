'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

from depth_funcs import (gen_usph_vecs,
                         # gen_usph_vecs_mp,
                         gen_usph_vecs_norm_dist)

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

    n_vecs = int(1e3)
    n_dims = 4

    os.chdir(main_dir)

    _strt = timeit.default_timer()
    gen_usph_vecs(n_vecs, n_dims)
    _end = timeit.default_timer()
    print('rand: %0.4f' % (_end - _strt))

    _strt = timeit.default_timer()
    gen_usph_vecs_norm_dist(n_vecs, n_dims)
    _end = timeit.default_timer()
    print('norm: %0.4f' % (_end - _strt))

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    if _save_log_:
        log_link.stop()
