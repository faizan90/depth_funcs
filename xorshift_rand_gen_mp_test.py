'''
Created on Nov 21, 2017

@author: Faizan-Uni

path: P:/Synchronize/IWS/2016_DFG_SPATE/scripts_p3/xorshift_rand_gen_test.py

'''

import timeit
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import pyximport
pyximport.install()

from test_rand_gen_mp import (gen_n_rands,
                              gen_n_rands_mp,
                              get_n_rands)

np.set_printoptions(precision=16,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.16f}'.format})

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    n_rands = 5000

#     gen_n_rands(n_rands)
#     gen_n_rands_mp(n_rands)

    randoms = get_n_rands(n_rands)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
               ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
