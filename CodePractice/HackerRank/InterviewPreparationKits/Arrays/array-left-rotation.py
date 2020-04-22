#3. https://www.hackerrank.com/challenges/array-left-rotation/problem
# !/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nd = input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = list(map(int, input().rstrip().split()))

    new_arr = [0 for _ in range(n)]

    for Index in range(n):
        newIndex = (Index - d ) % n
        new_arr[newIndex] = a[Index]

    print(' '.join(map(str, new_arr)))




