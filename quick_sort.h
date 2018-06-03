/*

Quick sort

Worst case performance: O(n^2)
Best case performance: O(n)
Average performance: O(n log(n))
Worst case space complexity: O(n)

*/

typedef double DT_D;
typedef long long DT_UL;


void quick_sort(DT_D arr[], DT_UL first_index, DT_UL last_index) {
  // declaring index variables
  DT_UL pivotIndex, index_a, index_b;
  DT_D temp;

  if (first_index < last_index) {
    // assigning first element index as pivot element
    pivotIndex = first_index;
    index_a = first_index;
    index_b = last_index;

    // Sorting in Ascending order with quick sort
    while (index_a < index_b) {
      while (arr[index_a] <= arr[pivotIndex] && index_a < last_index) {
        index_a++;
      }
      while (arr[index_b] > arr[pivotIndex]) {
        index_b--;
      }

      if (index_a < index_b) {
        // Swapping operation
        temp = arr[index_a];
        arr[index_a] = arr[index_b];
        arr[index_b] = temp;
      }
    }

    // At the end of first iteration, swap pivot element with index_b element
    temp = arr[pivotIndex];
    arr[pivotIndex] = arr[index_b];
    arr[index_b] = temp;

    // Recursive call for quick sort, with partitioning
    quick_sort(arr, first_index, index_b - 1);
    quick_sort(arr, index_b + 1, last_index);
  }
}
