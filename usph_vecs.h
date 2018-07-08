#pragma once
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <Windows.h>
#define WIN32_LEAN_AND_MEAN
#include <stdint.h>

#include "rand_gen_mp.h"

double log(double x);
double pow(double x, double y);
double rand_c_mp(unsigned long long &seed);


int gettimeofday(struct timeval *tp)
{
	// from https://stackoverflow.com/questions/10905892/equivalent-of-gettimeday-for-windows
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}


double usph_norm_ppf_c(double p) {
	double t, z;

	if (p <= 0.0) {
		return -INFINITY;
	}
	else if (p >= 1.0) {
		return INFINITY;
	}

	if (p > 0.5) {
		t = pow(-2.0 * log(1 - p), 0.5);
	}
	else {
		t = pow(-2.0 * log(p), 0.5);
	}

    z = -0.322232431088 + t * (-1.0 + t * (-0.342242088547 + t * (
        (-0.020423120245 + t * -0.453642210148e-4))));
    z = z / (0.0993484626060 + t * (0.588581570495 + t * (
        (0.531103462366 + t * (0.103537752850 + t * 0.3856070063e-2)))));
    z = z + t;

    if (p < 0.5) {
        z = -z;
    }

    return z;
}


void gen_usph_vecs_norm_dist_c(
		unsigned long long *seeds_arr,
		double *rn_ct_arr,
		double *ndim_usph_vecs,
		long n_vecs,
		long n_dims,
		long n_cpus) {

	size_t j, tid;
	timeval t0;
	long long i;
	unsigned long re_seed_i = (unsigned long) (1e6);

	for (tid = 0; tid < n_cpus; ++tid) {
		gettimeofday(&t0);
//		printf("t0: %d, before: %llu, ", t0.tv_usec, seeds_arr[tid]);
		seeds_arr[tid] = (unsigned long long) (t0.tv_usec * (long) 324234543);

		for (j = 0; j < 1000; ++j) {
			rand_c_mp(&seeds_arr[tid]);
		}
//		printf("after: %llu\n", seeds_arr[tid]);
		Sleep(45);
	}

	omp_set_num_threads(n_cpus);

	#pragma omp parallel for private(j)
	for (i = 0; i < n_vecs; ++i) {
		tid = omp_get_thread_num();
		double mag = 0.0;

		double *usph_vec = &ndim_usph_vecs[i * n_dims];
		for (j = 0; j < n_dims; ++j) {
			usph_vec[j] = usph_norm_ppf_c(rand_c_mp(&seeds_arr[tid]));
			mag = mag + pow(usph_vec[j], 2.0);
		}

		mag = pow(mag, 0.5);

		for (j = 0; j < n_dims; ++j) {
			usph_vec[j] = usph_vec[j]  / mag;
		}

		rn_ct_arr[tid] = rn_ct_arr[tid] + n_dims;
		if ((rn_ct_arr[tid]  / re_seed_i) > 1) {
			seeds_arr[tid] = (unsigned long long) (t0.tv_usec * (long) 937212);
			for (j = 0; j < 1000; ++j) {
				rand_c_mp(&seeds_arr[tid]);
			}

			Sleep(35);
			rn_ct_arr[tid] = 0.0;
		}
	}
}
