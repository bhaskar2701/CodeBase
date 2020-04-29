import sys

def slidingWindowBruteForce(arr, window):
    n = len(arr)
    max_sum = -float("inf")
    for i in range(n-window + 1):
        current_sum = 0
        for j in range(i, i+ window):
            current_sum += arr[j]

        if current_sum > max_sum:
            max_sum = current_sum

    return max_sum


def slidingWindow(arr, window):
    max_sum = sum(arr[i] for i in range(window))
    n = len(arr)
    for i in range(1, n-window+1):
        max_sum = max(max_sum, max_sum - arr[i-1] + arr[i + window -1])

    return  max_sum

if __name__ == "__main__":
    array = [1 ,2, 3, 4, 5, 6, 7]
    k = 4
    max_sum1 = slidingWindowBruteForce(array, k)
    if max_sum1:
        print("value of max sum: ", max_sum1 )
    max_sum2 = slidingWindow(array, k)
    if max_sum2:
        print("Value of max sum: ", max_sum2 )