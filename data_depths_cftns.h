#pragma once
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "quick_sort.h"
#include "searchsorted.h"

void depth_ftn_c_f64(
	const double *ref,
	const double *test,
	const double *uvecs,
		  double *dot_ref,
		  double *dot_test,
		  double *dot_test_sort,
		  long long *temp_mins,
		  long long *mins,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims,
	const long n_cpus);


void depth_ftn_c_f32(
	const float *ref,
	const float *test,
	const float *uvecs,
		  float *dot_ref,
		  float *dot_test,
		  float *dot_test_sort,
		  long *temp_mins,
		  long *mins,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims,
	const long n_cpus);


void depth_ftn_c_gf32(
	const float *ref,
	const float *test,
	const float *uvecs,
		  long *depths,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims);


void depth_ftn_c_f64(
	const double *ref,
	const double *test,
	const double *uvecs,
		  double *dot_ref,
		  double *dot_test,
		  double *dot_test_sort,
		  long long *temp_mins,
		  long long *mins,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims,
	const long n_cpus) {

	size_t tid;
	long long i;
	double _inc_mult = (double) (1 - (double) (1e-7));

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < n_uvecs; ++i) {

		size_t j, k;
		double *uvec, *sdot_ref, *sdot_test, *sdot_test_sort;
		long long *stemp_mins, *smins, _idx;
		double stest_med;

		tid = omp_get_thread_num();

		uvec = (double *) &uvecs[i * n_dims];

		sdot_ref = &dot_ref[tid * n_ref];
		for (j = 0; j < n_ref; ++j) {
			sdot_ref[j] = 0.0;
			for (k = 0; k < n_dims; ++k) {
				sdot_ref[j] = sdot_ref[j] + (uvec[k] * ref[(j * n_dims) + k]);
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
		}

		quick_sort_f64(&sdot_ref[0], 0, n_ref - 1);
		quick_sort_f64(&sdot_test_sort[0], 0, n_test - 1);

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
			stemp_mins[j] = searchsorted_f64(&sdot_ref[0], sdot_test[j], n_ref);
		}

		for (j = 0; j < n_test; ++j) {
			_idx = n_ref - stemp_mins[j];

			if (_idx < stemp_mins[j]) {
				stemp_mins[j] = _idx;
			}

			if (stemp_mins[j] < smins[j]) {
				smins[j] = stemp_mins[j];
			}
		}
	}
	return;
}

void depth_ftn_c_f32(
	const float *ref,
	const float *test,
	const float *uvecs,
		  float *dot_ref,
		  float *dot_test,
		  float *dot_test_sort,
		  long *temp_mins,
		  long *mins,
	const long n_ref,
	const long n_test,
	const long n_uvecs,
	const long n_dims,
	const long n_cpus) {

	size_t tid;
	long i;
	float _inc_mult = (float) (1 - (float) (1e-7));

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < n_uvecs; ++i) {

		size_t j, k;
		float *uvec, *sdot_ref, *sdot_test, *sdot_test_sort;
		long *stemp_mins, *smins, _idx;
		float stest_med;

		tid = omp_get_thread_num();

		uvec = (float *) &uvecs[i * n_dims];

		sdot_ref = &dot_ref[tid * n_ref];
		for (j = 0; j < n_ref; ++j) {
			sdot_ref[j] = 0.0;
			for (k = 0; k < n_dims; ++k) {
				sdot_ref[j] = sdot_ref[j] + (uvec[k] * ref[(j * n_dims) + k]);
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
		}

		quick_sort_f32(&sdot_ref[0], 0, n_ref - 1);
		quick_sort_f32(&sdot_test_sort[0], 0, n_test - 1);

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
			stemp_mins[j] = searchsorted_f32(&sdot_ref[0], sdot_test[j], n_ref);
		}

		for (j = 0; j < n_test; ++j) {
			_idx = n_ref - stemp_mins[j];

			if (_idx < stemp_mins[j]) {
				stemp_mins[j] = _idx;
			}

			if (stemp_mins[j] < smins[j]) {
				smins[j] = stemp_mins[j];
			}
		}
	}
	return;
}
