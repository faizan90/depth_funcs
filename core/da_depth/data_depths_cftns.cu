#pragma once
#include <stdio.h>
#include <time.h>
//#include <helper_cuda.h>

#define  MAX_LEVELS  300


int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta
      if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n");
      break;
      }
    return cores;
}



__device__ void quick_sort_gf32(
		float *arr,
		long first_index,
		long last_index) {

	// declaring index variables
	long pivotIndex, index_a, index_b;
	float temp;

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
		quick_sort_gf32(arr, first_index, index_b - 1);
		quick_sort_gf32(arr, index_b + 1, last_index);
	}
	return;
}


/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */

__device__ void merge(float *arr, long l, long m, long r)
{
    long i, j, k;
    long n1 = m - l + 1;
    long n2 =  r - m;

    /* create temp arrays */
    float *L = new float[n1], *R = new float[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}


/* Iterative mergesort function to sort arr[0...n-1] */

__device__ void mergeSort(float *arr, long n)
{
   long curr_size;  // For current size of subarrays to be merged
                   // curr_size varies from 1 to n/2
   long left_start; // For picking starting index of left subarray
                   // to be merged

   // Merge subarrays in bottom up manner.  First merge subarrays of
   // size 1 to create sorted subarrays of size 2, then merge subarrays
   // of size 2 to create sorted subarrays of size 4, and so on.
   for (curr_size=1; curr_size<=n-1; curr_size = 2*curr_size)
   {
       // Pick starting point of different subarrays of current size
       for (left_start=0; left_start<n-1; left_start += 2*curr_size)
       {
           // Find ending point of left subarray. mid+1 is starting
           // point of right
           int mid = left_start + curr_size - 1;

           int right_end = min(left_start + 2*curr_size - 1, n-1);

           // Merge Subarrays arr[left_start...mid] & arr[mid+1...right_end]
           merge(arr, left_start, mid, right_end);
       }
   }
}



__device__ void quickSort(float *arr, long elements) {

  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, swap;
  float piv;

  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      piv=arr[L];
      while (L<R) {
        while (arr[R]>=piv && L<R) R--; if (L<R) arr[L++]=arr[R];
        while (arr[L]<=piv && L<R) L++; if (L<R) arr[R--]=arr[L]; }
      arr[L]=piv; beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap; }}
    else {
      i--; }}}



__device__ long searchsorted_gf32(
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
//			printf("%d, %d, %d, %f\n", first, last, curr_idx, value);
			return curr_idx;
		}
	}
	return 0;
}


__global__ void fill_smins_gf32(
	const float *uvecs,
	const float *ref,
	const float *test,
		  float *dot_ref,
		  float *dot_test,
		  float *dot_test_sort,
		  long *mins,
		  long *temp_mins,
	const long n_uvecs,
	const long n_ref,
	const long n_test,
	const long n_dims) {

	size_t tid;
	tid = ((blockIdx.x * blockDim.x) +
		   threadIdx.x);

	if (tid >= n_uvecs) {
		return;
	}

//	printf("tid: %d\n", tid);
	float _inc_mult = (float) (1 - (float) (1e-7));

	size_t i, j, k;
	float *uvec, *sdot_ref, *sdot_test, *sdot_test_sort;
	long *stemp_mins, *smins, _idx;
	float stest_med;

	i = blockIdx.x;
	uvec = (float *) &uvecs[i * n_dims];

	sdot_ref = &dot_ref[tid * n_ref];
	for (j = 0; j < n_ref; ++j) {
		sdot_ref[j] = 0.0;
		for (k = 0; k < n_dims; ++k) {
			sdot_ref[j] = sdot_ref[j] + (uvec[k] * ref[(j * n_dims) + k]);
//			printf("sdot_ref[j]: %0.5f\n", sdot_ref[j]);
		}
	}

	sdot_test = &dot_test[tid * n_test];
	sdot_test_sort = &dot_test_sort[tid * n_test];
	for (j = 0; j < n_test; ++j) {
		sdot_test[j] = 0.0;
		for (k = 0; k < n_dims; ++k) {
			sdot_test[j] = sdot_test[j] + (
					uvec[k] * test[(j * n_dims) + k]);
		}
		sdot_test_sort[j] = sdot_test[j];
//		printf("sdot_test[j]: %0.5f\n", sdot_test[j]);
	}

	quick_sort_gf32(&sdot_ref[0], 0, n_ref - 1);
//	mergeSort(&sdot_ref[0], n_ref);
	quick_sort_gf32(&sdot_test_sort[0], 0, n_test - 1);
//	quickSort(&sdot_ref[0], n_ref);
////	quickSort(&sdot_test_sort[0], n_test);
//
	if ((n_test % 2) == 0) {
		stest_med = 0.5 * (sdot_test_sort[n_test / 2] +
						   sdot_test_sort[(n_test / 2) - 1]);
	}

	else {
		stest_med = sdot_test_sort[n_test / 2];
	}

	for (j = 0; j < n_test; ++j) {
		sdot_test[j] = (
				(sdot_test[j] - stest_med) * _inc_mult) + stest_med;
	}

	smins = &mins[tid * n_test];
	stemp_mins = &temp_mins[tid * n_test];

	for (j = 0; j < n_test; ++j) {
		stemp_mins[j] = searchsorted_gf32(&sdot_ref[0], sdot_test[j], n_ref);
//		printf("sdot_ref[0], stemp_mins[j]: %0.5f, %d\n", sdot_ref[0], stemp_mins[j]);
	}

	for (j = 0; j < n_test; ++j) {
		_idx = n_ref - stemp_mins[j];

		if (_idx < stemp_mins[j]) {
			stemp_mins[j] = _idx;
		}

		if (stemp_mins[j] < smins[j]) {
			smins[j] = stemp_mins[j];
		}
//		printf("smins[j]: %d\n", smins[j]);
	}

	return;
}

__global__ void test_ftn() {
	printf("345435345\n");
	return;
}

void depth_ftn_c_gf32(
	const float *ref,
	const float *test,
	const float *uvecs,
		  long *depths,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims) {

	size_t i, j, k;
//	int dev_ct;

//	cudaGetDeviceCount(&dev_ct);
//	printf("GPU count is: %d\n", dev_ct);

	float *d_uvecs, *d_ref, *d_test, *dot_ref, *dot_test, *dot_test_sort;
	long *mins, *temp_mins;

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

//	int nDevices;
//
//	cudaGetDeviceCount(&nDevices);
//	for (i = 0; i < nDevices; i++) {
//		cudaDeviceProp prop;
//		cudaGetDeviceProperties(&prop, i);
//		printf("Device Number: %zd\n", i);
//		printf("  Device name: %s\n", prop.name);
//		printf("  Memory Clock Rate (KHz): %d\n",
//			   prop.memoryClockRate);
//		printf("  Memory Bus Width (bits): %d\n",
//			   prop.memoryBusWidth);
//		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
//			   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
//		}

	int n_cpus = n_uvecs; //getSPcores(deviceProp);
	printf("n_cpus: %d\n", n_cpus);
	printf("n_uvecs: %d\n", n_uvecs);
	printf("n_ref: %d\n", n_ref);
	printf("n_test: %d\n", n_test);
	printf("n_dims: %d\n", n_dims);

	cudaError_t error;

//	printf("1\n");
	cudaMallocManaged(&d_uvecs, (n_uvecs * n_dims) * sizeof(float));
	cudaMallocManaged(&d_ref, (n_ref * n_dims) * sizeof(float));
	cudaMallocManaged(&d_test, (n_test * n_dims) * sizeof(float));

	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error 1: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaMallocManaged(&dot_ref, (n_cpus * n_ref) * sizeof(float));
	cudaMallocManaged(&dot_test, (n_cpus * n_test) * sizeof(float));
	cudaMallocManaged(&dot_test_sort, (n_cpus * n_test) * sizeof(float));

	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error 2: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	cudaMallocManaged(&mins, (n_cpus * n_test) * sizeof(long));
	cudaMallocManaged(&temp_mins, (n_cpus * n_test) * sizeof(long));
//	printf("n_test: %d\n", n_test);

	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error 3: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	for (i = 0; i < (n_ref * n_dims); ++i) {
		d_ref[i] = ref[i];
	}

	for (i = 0; i < (n_test * n_dims); ++i) {
		d_test[i] = test[i];
	}

	for (i = 0; i < (n_uvecs * n_dims); ++i) {
		d_uvecs[i] = uvecs[i];
	}

	for (i = 0; i < (n_uvecs * n_test); ++i) {
		temp_mins[i] = n_ref;
		mins[i] = n_ref;

	}
	printf("0, mins: %d\n", mins[0]);

	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error 4: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	//	printf("\n\n");

//	clock_t start = clock(), diff;

//	printf("2\n");

	cudaDeviceSetLimit(cudaLimitStackSize, 200 * 1024);

//	for (i = 0; i < n_uvecs; ++i) {
//		fill_smins_gf32(
//				i,
//				d_uvecs,
//				d_ref,
//				d_test,
//				dot_ref,
//				dot_test,
//				dot_test_sort,
//				mins,
//				temp_mins,
//				n_uvecs,
//				n_ref,
//				n_test,
//				n_dims);
//	}

	dim3 block_size(1024);
	dim3 thread_size(32);

	fill_smins_gf32 <<< block_size, thread_size >>> (
		d_uvecs,
		d_ref,
		d_test,
		dot_ref,
		dot_test,
		dot_test_sort,
		mins,
		temp_mins,
		n_uvecs,
		n_ref,
		n_test,
		n_dims);
//	test_ftn <<< 1, 1 >>> ();

	cudaDeviceSynchronize();
//	printf("3\n");

	error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error 5: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

	for (k = 0; k < n_test; ++k) {
		for (j = 0; j < n_cpus; ++j) {
//			printf("k: %zu, j: %zu, mins: %d\n", k, j, mins[j * (n_test) + k]);
			if (depths[k] > mins[j * (n_test) + k]) {
				depths[k] = mins[j * (n_test) + k];
//				printf("Updated depths[k]: %zu, %d\n", k, depths[k]);

//				if (!depths[k]) {
//					break;
//				}
			}
		}
//		printf("\n\n");
	}

//	for (i = 0; i < n_test; ++i) {
//		printf("depths[i]: %d\n", depths[i]);
//	}

//	printf("4\n");
//	diff = clock() - start;
//	int msec = diff * 1000 / CLOCKS_PER_SEC;
//	printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);

	return;
}
