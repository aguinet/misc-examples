#include <iostream>
#include <vector>

#include <omp.h>
#include <string.h>
#include <math.h>

#include "bench.h"
#include "compute.h"

uint32_t* compute(float const* pts, const size_t n, const uint32_t size_interval_u32, const float size_interval)
{
	std::vector<uint32_t*> hist_locals;
	hist_locals.resize(omp_get_num_threads(), NULL);

	BENCH_START(b);

#pragma omp parallel num_threads(4)
	{
		uint32_t* hist_local;
		posix_memalign((void**) &hist_local, 16, sizeof(uint32_t)*size_interval_u32);
		memset(hist_local, 0, sizeof(uint32_t)*size_interval_u32);
		hist_locals[omp_get_thread_num()] = hist_local;

#pragma omp for
		for (size_t i = 0; i < n; i++) {
			const size_t idx = floor(pts[i]/size_interval);
			hist_local[idx]++;
		}
	}

	std::cerr << "Final reduction..." << std::endl;
	uint32_t* final_hist = NULL;

	// Find the first non-null partial histogram
	std::vector<uint32_t*>::const_iterator it;
	for (it = hist_locals.begin(); it != hist_locals.end(); it++) {
		if (*it != NULL) {
			final_hist = *it;
			it++;
			break;
		}
	}

	if (final_hist == NULL) {
		return NULL;
	}

	// Final reduction
	for (; it != hist_locals.end(); it++) {
		uint32_t* hist_local = *it;
		if (hist_local == NULL) {
			continue;
		}
		for (uint32_t i = 0; i < size_interval_u32; i++) {
			final_hist[i] += hist_local[i];
		}
	}

	BENCH_END(b, "compute_omp", sizeof(float), n, sizeof(uint32_t), n);

	return final_hist;
}
