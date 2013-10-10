#include <iostream>
#include <vector>

#include <omp.h>
#include <string.h>
#include <math.h>

#include "compute.h"
#include "bench.h"

uint32_t* compute(float const* pts, const size_t n, const uint32_t size_hist, const float size_interval)
{
	uint32_t* hist = (uint32_t*) malloc(sizeof(uint32_t)*size_hist);

	BENCH_START(b);
#pragma omp parallel for
	for (size_t i = 0; i < n; i++) {
		const size_t idx = floor(pts[i]/size_interval);
		hist[idx]++;
	}
	BENCH_END(b, "compute", sizeof(float), n, sizeof(uint32_t), n);

	return hist;
}
