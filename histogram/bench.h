#ifndef BENCH_H
#define BENCH_H

#include "helpers.h"

#define BENCH_START(var)\
	double __bench_start_##var = get_current_timestamp();

#define BENCH_END_NODISP(var)\
	double __bench_end_##var = get_current_timestamp();

#define BENCH_END(var, desc, nelts_start, size_elt_start, nelts_end, size_elt_end)\
	BENCH_END_NODISP(var);\
	{\
		double time = (__bench_end_##var-__bench_start_##var);\
		double size_in_mb = (double)((size_t)nelts_start*(size_t)size_elt_start)/(1024.0*1024.0);\
		double size_out_mb = (double)((size_t)nelts_end*(size_t)size_elt_end)/(1024.0*1024.0);\
		double bw_in = size_in_mb/time;\
		double bw_out = size_out_mb/time;\
		fprintf(stderr, "%s: in %0.5f ms. Input (#/size/BW): %lu/%0.5f MB/%0.5f MB/s | Output (#/size/BW): %lu/%0.5f MB/%0.5f MB/s\n",\
		       desc, time*1000.0, (size_t) nelts_start, size_in_mb, bw_in, (size_t) nelts_end, size_out_mb, bw_out);\
	}

#define BENCH_END_TIME(var)\
		((__bench_end_##var-__bench_start_##var))

#define BENCH_END_SAME_TYPE(var, desc, nelts_in, nelts_out, size_elt) BENCH_END(var, desc, nelts_in, size_elt, nelts_out, size_elt)
#define BENCH_END_TRANSFORM(var, desc, nelts, size_elt) BENCH_END_SAME_TYPE(var, desc, nelts, nelts, size_elt)

#ifdef __cplusplus
#define CHECK(v) __CHECK(v, __FILE__, __LINE__)
#define __CHECK(v,F,L)\
	if (!(v)) {\
		std::cerr << F << ":" << L << " :" << #v << " isn't valid." << std::endl;\
	}
#endif

#endif
