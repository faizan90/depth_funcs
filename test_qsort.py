'''
@author: Faizan-Uni-Stuttgart

1 Jul 2020

09:26:09

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from depth_funcs import sort_arr

plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    n_vals = int(1e6)
    n_iters = 1

    invld_ctr = 0

    i = 0
    while (i < n_iters):

#         rand_vals_ref = -100 + ((200) * np.random.random(n_vals))
#         rand_vals_ref = np.zeros(n_vals)
        rand_vals_ref = np.arange(n_vals).astype(np.float64)

#         rand_vals_ref = np.random.randint(
#             0, n_vals * 2, n_vals).astype(np.float64)

        rand_vals_sim = rand_vals_ref.copy()

        beg_time = timeit.default_timer()
        rand_vals_ref.sort()
        end_time = timeit.default_timer()

        ref_time = end_time - beg_time

        beg_time = timeit.default_timer()
        sort_arr(rand_vals_sim)
        end_time = timeit.default_timer()

        sim_time = end_time - beg_time

        ref_sim_diffs = rand_vals_ref - rand_vals_sim

        ref_sim_min_diff = ref_sim_diffs.min()
        ref_sim_max_diff = ref_sim_diffs.max()

        ref_ref_diffs = rand_vals_ref[1:] - rand_vals_ref[:-1]

        ref_ref_min_diff = ref_ref_diffs.min()
        ref_ref_max_diff = ref_ref_diffs.max()

        sim_sim_diffs = rand_vals_sim[1:] - rand_vals_sim[:-1]

        sim_sim_min_diff = sim_sim_diffs.min()
        sim_sim_max_diff = sim_sim_diffs.max()

        if ((sim_sim_min_diff < 0) or
            (ref_sim_min_diff != 0) or
            (ref_sim_max_diff != 0)):

            invld_ctr += 1

#             tre = 1
#
#             plt.plot(ref_sim_diffs, alpha=0.7)
#             plt.grid()
#             plt.show()
#             plt.close()

        print('\n')
        print('#' * 30)
        print('Iteration:', i)

        print('\n')
        print(f'Ref. time: {ref_time:0.3f}')
        print(f'Sim. time: {sim_time:0.3f}')

        print('\n')
        print(
            'Min. and max. ref_sim_diff:', ref_sim_min_diff, ref_sim_max_diff)

        print('\n')
        print(
            'Min. and max. ref_ref_diff:', ref_ref_min_diff, ref_ref_max_diff)

        print('\n')
        print(
            'Min. and max. sim_sim_diff:', sim_sim_min_diff, sim_sim_max_diff)

        i += 1

    print('\n')
    print('#' * 30)
    print('invld_ctr:', invld_ctr)

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
