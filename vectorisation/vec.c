#include <stdio.h>
#include <stdlib.h>

#include <x86intrin.h>
#include <math.h>
#include <time.h>

#include "bench.h"

void compute(float* r, float const* a, float const* b, float const* c, const size_t n)
{
	for (size_t i = 0; i < n; i++) {
		r[i] = sqrt(a[i]*b[i]+c[i]);
	}
}

void compute_sse(float* r, float const* a, float const* b, float const* c, const size_t n)
{
	const size_t nsse = (n/4)*4;
	__m128 sse_a, sse_b, sse_c, sse_r;
	for (size_t i = 0; i < nsse; i += 4) {
		sse_a = _mm_load_ps(&a[i]);
		sse_b = _mm_load_ps(&b[i]);
		sse_c = _mm_load_ps(&c[i]);
		sse_r = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(sse_a, sse_b), sse_c));
		_mm_store_ps(&r[i], sse_r);
	}
	// Prologue
	for (size_t i = nsse; i < n; i++) {
		r[i] = sqrt(a[i]*b[i]+r[i]);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s n\n", argv[0]);
		return 1;
	}

	const size_t n = atoll(argv[1]);
	float *a, *b, *c, *r;
	posix_memalign((void**) &a, 16, sizeof(float)*n);
	posix_memalign((void**) &b, 16, sizeof(float)*n);
	posix_memalign((void**) &c, 16, sizeof(float)*n);
	posix_memalign((void**) &r, 16, sizeof(float)*n);

	srand(time(NULL));
	for (size_t i = 0; i < n; i++) {
		a[i] = i;
		b[i] = i/2;
		c[i] = i*2;
	}

	compute(r, a, b, c, n);

	BENCH_START(serial);
	compute(r, a, b, c, n);
	BENCH_END(serial, "serial", sizeof(float), n*3, sizeof(float), n);

	BENCH_START(sse);
	compute_sse(r, a, b, c, n);
	BENCH_END(sse, "sse", sizeof(float), n*3, sizeof(float), n);

	return 0;
}
