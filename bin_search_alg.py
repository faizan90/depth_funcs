'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np

import pyximport

pyximport.install()

from depth_cy_ftns import searchsorted_cy

# iterative implementation of binary search in Python


def binary_search(a_list, item):
    """Performs iterative binary search to find the position of an integer in a given, sorted, list.
    a_list -- sorted list of integers
    item -- integer you are searching for the position of
    """

    first = 0
    last = len(a_list) - 1

    if item <= a_list[0]:
        print('{item} insert at position {i}'.format(item=item, i=0))
        return

    elif item > a_list[-1]:
        print('{item} insert at position {i}'.format(item=item, i=(last + 1)))
        return

    while first <= last:
        i = (first + last) // 2

        if a_list[i] < item <= a_list[i + 1]:
            print('{item} insert at position {i}'.format(item=item, i=i + 1))
            return
        elif a_list[i] > item:
            last = i - 1
        elif a_list[i] < item:
            first = i + 1
        else:
#             print('{item} not found in the list'.format(item=item))
            print('{item} insert at position {i}'.format(item=item, i=first + 1))
            return

    return


def binary_search_tol(a_list, item):
    """Performs iterative binary search to find the position of an integer in a given, sorted, list.
    a_list -- sorted list of integers
    item -- integer you are searching for the position of
    """

    first = 0
    last = len(a_list) - 1

    if item <= a_list[0]:
        print('{item} insert at position {i}'.format(item=item, i=0))
        return

    elif item > a_list[-1]:
        print('{item} insert at position {i}'.format(item=item, i=(last + 1)))
        return

    while first <= last:
        i = (first + last) // 2

        if a_list[i] < item <= a_list[i + 1]:
            print('{item} insert at position {i}'.format(item=item, i=i + 1))
            return
        elif a_list[i] > item:
            last = i - 1
        elif a_list[i] < item:
            first = i + 1
        else:
#             print('{item} not found in the list'.format(item=item))
            print('{item} insert at position {i}'.format(item=item, i=first + 1))
            return

    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    long_list = [0, 0.02, 0.02, 0.02, 0.02, 4, 10, 25]

    short_list = [-10, 0.01, 0, 22, 23, 4, 26]

    os.chdir(main_dir)

    long_list.sort()

    for short in short_list:
        binary_search(long_list, short)

    print(np.searchsorted(long_list, short_list))
    print(searchsorted_cy(np.array(long_list, dtype=float),
                          np.array(short_list, dtype=float)))

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
