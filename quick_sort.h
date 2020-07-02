/*

Quick sort

Worst case performance: O(n^2)
Best case performance: O(n)
Average performance: O(n log(n))
Worst case space complexity: O(n)

NOTE: The older algorithm is only good for a sequence that is randomly shuffled.
For cases such as all values similar and sorted or reverse sorted arrays, it
takes forever. The newe Hoar algorithm performs better on sorted or similar
arrays. Takes a bit more time than numpy.

*/

#pragma once


void quick_sort_f64(double *arr, long long first_index, long long last_index) {

	// Hoare Impl at:
	// https://stackoverflow.com/questions/33837737/quick-sort-middle-pivot-implementation-strange-behaviour

	if (first_index >= last_index) return;

    double pivot = arr[first_index + ((last_index - first_index) / 2)];

    long long index_a = first_index - 1;

    long long index_b = last_index + 1;

    double temp;

    while (1) {

    	while(arr[++index_a] < pivot);

        while(arr[--index_b] > pivot);

        if (index_a >= index_b) break;

        temp = arr[index_a];
        arr[index_a] = arr[index_b];
        arr[index_b] = temp;
    }

    quick_sort_f64(arr, first_index, index_b);
    quick_sort_f64(arr, index_b + 1, last_index);

    // Older implementation
//	long long pivotIndex, index_a, index_b;
//	double temp;
//
//	if (first_index < last_index) {
//
//		// assigning first element index as pivot element
//		pivotIndex = first_index + ((last_index - first_index) / 2);
//		index_a = first_index;
//		index_b = last_index;
//
//		// Sorting in Ascending order with quick sort
//		while (index_a < index_b) {
//
//			while ((arr[index_a] <= arr[pivotIndex]) &&
//				   (index_a < last_index)) {
//
//				index_a++;
//			}
//
//			while (arr[index_b] > arr[pivotIndex]) {
//				index_b--;
//			}
//
//			if (index_a < index_b) {
//			// Swapping operation
//				temp = arr[index_a];
//				arr[index_a] = arr[index_b];
//				arr[index_b] = temp;
//			}
//
//		}
//
//		// At the end of first iteration,
//		// swap pivot element with index_b element
//		temp = arr[pivotIndex];
//		arr[pivotIndex] = arr[index_b];
//		arr[index_b] = temp;
//
//		quick_sort_f64(arr, first_index, index_b - 1);
//		quick_sort_f64(arr, index_b + 1, last_index);
//	}
	return;
}


void quick_sort_f32(float *arr, long first_index, long last_index) {

	if (first_index >= last_index) return;

    float pivot = arr[first_index + ((last_index - first_index) / 2)];

    long index_a = first_index - 1;

    long index_b = last_index + 1;

    float temp;

    while (1) {

    	while(arr[++index_a] < pivot);

        while(arr[--index_b] > pivot);

        if (index_a >= index_b) break;

        temp = arr[index_a];
        arr[index_a] = arr[index_b];
        arr[index_b] = temp;
    }

    quick_sort_f32(arr, first_index, index_b);
    quick_sort_f32(arr, index_b + 1, last_index);

//	// declaring index variables
//	long pivotIndex, index_a, index_b;
//	float temp;
//
//	if (first_index < last_index) {
//		// assigning first element index as pivot element
//		pivotIndex = first_index;
//		index_a = first_index;
//		index_b = last_index;
//
//		// Sorting in Ascending order with quick sort
//		while (index_a < index_b) {
//			while (arr[index_a] <= arr[pivotIndex] && index_a < last_index) {
//				index_a++;
//			}
//			while (arr[index_b] > arr[pivotIndex]) {
//				index_b--;
//			}
//
//			if (index_a < index_b) {
//			// Swapping operation
//				temp = arr[index_a];
//				arr[index_a] = arr[index_b];
//				arr[index_b] = temp;
//			}
//		}
//
//		// At the end of first iteration,
//		// swap pivot element with index_b element
//		temp = arr[pivotIndex];
//		arr[pivotIndex] = arr[index_b];
//		arr[index_b] = temp;
//
//		// Recursive call for quick sort, with partitioning
//		quick_sort_f32(arr, first_index, index_b - 1);
//		quick_sort_f32(arr, index_b + 1, last_index);
//	}
	return;
}

