//#include <stdio.h>
#pragma once


long long searchsorted_f64(
		const double *arr,
		const double value,
		const long long arr_size) {

	// arr must be sorted
	long long first = 0, last = arr_size - 1, curr_idx;

	if (value <= arr[0]) {
		return 0;
	}

	else if (value > arr[last]) {
		return arr_size;
	}

	while (first <= last) {
		curr_idx = (long long) (0.5 * (first + last));
		if ((value > arr[curr_idx]) && (value <= arr[curr_idx + 1])) {
			return curr_idx + 1;
		}

		else if (value < arr[curr_idx]) {
			last = curr_idx - 1;
		}

		else if (value > arr[curr_idx]) {
			first = curr_idx + 1;
		}

		else {
			//printf("%d, %d, %d, %f\n", first, last, curr_idx, value);
			return curr_idx;
		}
	}
	return 0;
}


long searchsorted_f32(
		const float *arr,
		const float value,
		const long arr_size) {

	// arr must be sorted
	long first = 0, last = arr_size - 1, curr_idx;

	if (value <= arr[0]) {
		return 0;
	}

	else if (value > arr[last]) {
		return arr_size;
	}

	while (first <= last) {
		curr_idx = (long) (0.5 * (first + last));
		if ((value > arr[curr_idx]) && (value <= arr[curr_idx + 1])) {
			return curr_idx + 1;
		}

		else if (value < arr[curr_idx]) {
			last = curr_idx - 1;
		}

		else if (value > arr[curr_idx]) {
			first = curr_idx + 1;
		}

		else {
			//printf("%d, %d, %d, %f\n", first, last, curr_idx, value);
			return curr_idx;
		}
	}
	return 0;
}
