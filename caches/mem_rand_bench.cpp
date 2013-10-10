#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <string.h>
#include <algorithm>

#include "bench.h"

#define NTRIES 20

double launch_bench(uint32_t const* idxes, const size_t nint, const size_t n, const int nthreads)
{
    double time_start, time_end;
#pragma omp parallel num_threads(nthreads)
    {
        uint32_t* buf = (uint32_t*) malloc(sizeof(uint32_t)*nint);
        memset(buf, 0, sizeof(uint32_t)*nint);

#pragma omp barrier
#pragma omp master
        {
            time_start = get_current_timestamp();
        }

        for (size_t i = 0; i < n; i++) {
            buf[idxes[i]] = 0;
        }

#pragma omp barrier

#pragma omp master
        {
            time_end = get_current_timestamp();
        }

        free(buf);
    }

    return ((double)(nthreads*sizeof(uint32_t)*n)/(time_end-time_start))/(1024.0*1024.0);
}

int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " size_buffer_uint number_ops nthreads" << std::endl;
		return 1;
	}

	const size_t nint = atoll(argv[1]);
	const size_t n = atoll(argv[2]);
    const int nthreads = atoi(argv[3]);

    srand(time(NULL));

    uint32_t* idxes = (uint32_t*) malloc(sizeof(uint32_t)*n);
    for (size_t i = 0; i < n; i++) {
        idxes[i] = rand()%nint;
    }

    double res[NTRIES];
    for (int i = 0; i < NTRIES; i++) {
        res[i] = launch_bench(idxes, nint, n, nthreads);
    }

    std::sort(&res[0], &res[NTRIES]);

    double mean = 0;
    for (int i = 5; i < NTRIES-5; i++) {
        mean += res[i];
    }
    mean /= NTRIES;

    std::cout << mean << std::endl;
    //std::cout << "BW: " << mean << " MB/s" << std::endl;

	return 0;
}
