#pragma once
#include <time.h>
#include <limits.h>
#include <stdlib.h>

typedef double DT_D;
typedef unsigned long DT_UL;
typedef unsigned long long DT_ULL;

const DT_D MOD_long_long = (DT_D) ULLONG_MAX;
static DT_ULL SEED = time(NULL) * 1000;
const DT_ULL A = 21, B = 35, C = 4;
static DT_ULL rnd_i = SEED;

DT_D rand_c() {
    rnd_i ^= rnd_i << A;
    rnd_i ^= rnd_i >> B;
    rnd_i ^= rnd_i << C;
    return rnd_i / MOD_long_long;
}


void warm_up() {
	DT_UL i;
	for (i=0; i<1000; ++i) {
	    //printf("%0.10f\n", rand_c());
	    rand_c();
	}
}


void iter_n_rands(DT_ULL n_iters) {
	DT_ULL i;
	
	for (i=n_iters; --i;) {
		rand_c();
	}

}


void re_seed(DT_ULL new_seed) {
	//printf("old SEED: %llu\n", SEED);
	
	SEED = new_seed;
	
	//printf("new SEED: %llu\n", SEED);
	
	//printf("old rnd_i: %llu\n", rnd_i);
	
	rnd_i = SEED;
	
	//printf("new rnd_i: %llu\n", rnd_i);
	
	warm_up();
}


//void main() {
	//printf("in main\n");
	//warm_up();
//}
