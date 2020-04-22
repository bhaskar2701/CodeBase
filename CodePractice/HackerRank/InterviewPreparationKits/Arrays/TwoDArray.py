#2. https://www.hackerrank.com/challenges/2d-array/problem

# !/bin/python3

import math
import os
import random
import re
import sys


# Complete the hourglassSum function below.
def hourglassSum(arr):
    rows = len(arr[0])
    cols = len(arr)

    if rows < 2 or cols < 2:
        return
    for row in range(rows - 2):
        for col in range(cols - 2):
            sum = arr[row][col] + arr[row][col + 1] + arr[row][col + 2] + arr[row + 1][col + 1] + arr[row + 2][col] + \
                  arr[row + 2][col + 1] + arr[row + 2][col + 2]
            if row == 0 and col == 0:
                maxsum = sum
            if sum > maxsum:
                maxsum = sum

    return maxsum


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)

    fptr.write(str(result) + '\n')

    fptr.close()


 # time complexity: O(m*n)
