'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.3f}'.format})

import pyximport
pyximport.install()

from depth_cy_ftns import gen_usph_vecs, gen_usph_vecs_mp

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    n_dims = 7
    n_vecs = int(1e8)
    n_cpus = 6

    os.chdir(main_dir)

#     usph_vecs = gen_usph_vecs(n_vecs, n_dims)
    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    mags = np.sqrt((usph_vecs ** 2).sum(axis=1))
    idxs = (mags > 1.001)
    print(idxs.sum(), np.where(idxs), mags[idxs])

    time.sleep(1000)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
