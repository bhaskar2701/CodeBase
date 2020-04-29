#Binary Search implementation.

array = [1, 2, 4, 5, 7, 9, 11, 12]
array1 = []
def binarySearch(arr , element):
    start = 0
    end = len(arr) -1


    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == element:
            return mid
        elif arr[mid] > element:
            end = mid -1
        else:
            start = end + 1


index = binarySearch(array1, 5)
print(index)