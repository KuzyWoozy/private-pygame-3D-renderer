import numpy as np

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, overload

from Exceptions import DoesNotInheritSortableException


class Sortable(ABC):
    """Children should override the sortBy method so that they can be used by the sorting algos in this package.""" 
    @abstractmethod
    def sortBy(self) -> float:
        """Return the value to sort by."""
        pass


def heapSort(arr: List[Any]) -> List[Any]:
    """
    Heapsort sorting algorithm :)

    Parameters: arr -- List of objects to sort, objects must inherit Sortable
    Returns: List of sorted objects (largest to smallest)
    """
    @overload
    def swap(index1: int, index2: int, arr: List[Any]) -> List[Any]:
        pass

    @overload
    def swap(index1: int, index2: int, arr: np.ndarray) -> np.ndarray:
        pass

    def swap(index1: int, index2: int, arr: Union[List[Any], np.ndarray]) -> Union[List[Any], np.ndarray]:
        """Swaps the two specified elements via index of the array."""
        temp = arr[index1]
        arr[index1] = arr[index2]
        arr[index2] = temp
        return arr



    def siftDown(i: int, arr_values: np.ndarray, arr_items: List[Any]) -> Tuple[np.ndarray, List[Any]]:
        """
        Moves the element down into its correct position in the heap.

        Parameters:
            i -- index of element to siftDown.
            arr_values -- Array of sortBy() values of the array we are sorting.
            arr_items -- Array of items we are sorting.
        
        Returns: tuple of updated arr's (arr_values, arr_items)
        """

        def bigger(index1: int, index2: int, arr: np.ndarray) -> int:
            """Returns the index of the bigger value in the array."""
            if (arr[index1] < arr[index2]):
                return index2
            else:
                return index1

        while(True):
            right_child = 2 * i + 1
            left_child = 2 * i + 2
            
            if (right_child >= arr_values.size and left_child >= arr_values.size):
                break
            elif (right_child >= arr_values.size):
                larger_index = left_child
            elif (left_child >= arr_values.size):
                larger_index = right_child
            else:
                larger_index = bigger(left_child, right_child, arr_values)

            if (arr_values[i] < arr_values[larger_index]):
                arr_values = swap(i, larger_index, arr_values)
                arr_items = swap(i, larger_index, arr_items)
                i = larger_index
            else:
                break

        return (arr_values, arr_items)

            
    array_length = len(arr)
    values = np.empty(array_length)
    
    # Construct arr_values and make sure all elements are sortable
    for i, item in enumerate(arr):
        if not isinstance(item, Sortable):
            raise DoesNotInheritSortableException
        values[i] = item.sortBy()

    # Enforce the heap property
    for item in range(array_length-1, -1, -1):
        (values, arr) = siftDown(item, values, arr)
    
    # Where sorting happens
    sorted_arr = []
    for item in range(1, array_length):
        sorted_arr.append(arr[0])
        arr = swap(0, array_length-item, arr)
        values = swap(0, array_length-item, values) 
        (values, arr) = siftDown(0, values[:-1], arr[:-1])

    sorted_arr.append(arr[0])

    return sorted_arr
    





    
        


    
