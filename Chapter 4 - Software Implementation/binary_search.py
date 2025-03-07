"""Implementation of Binary Search using Divide & Conquer Algorithm"""

from logzero import logger

def binary_search(arr: list, search_key: int, sort_orientation: str = "asc"):
    """Binary Search implementation

    Arguments:
        arr -- .
        search_key -- .
    """
    
    start, end = 0, len(arr)
    while start < end:
        mid = (start + end)//2

        if arr[mid] == search_key:
            logger.info(f"{search_key} found at index: {mid}")
            return
        elif ( sort_orientation == "asc" and search_key < arr[mid] ) or ( sort_orientation == "desc" and search_key > arr[mid] ):
            end = mid - 1
        else:
            start = mid + 1
    else:
        logger.info(f"{search_key} not found in the array")
    
if __name__ == "__main__":
    arr, key = sorted([1, 7, 8, 3, 2, 5]), 5 
    binary_search(arr, key)

