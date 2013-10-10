#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <cstdint>
#include <cstdlib>

#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_init.h>

#define L1_SIZE 512 // in bytes
#define L1_SIZE_DBLE (L1_SIZE/sizeof(double))
#define SIZE_INTERVAL (1.0/(L1_SIZE_DBLE))

template <typename T>
static inline T clamp(T const v, T const min, T const max)
{
	return std::max(std::min(v, max), min);
}

static inline double* init_tls()
{
	double* ret;
	posix_memalign((void**) &ret, 16, sizeof(double)*L1_SIZE_DBLE);
	memset(ret, 0, sizeof(double)*L1_SIZE_DBLE);
	return ret;
}

typedef tbb::enumerable_thread_specific<double*, tbb::cache_aligned_allocator<double>, tbb::ets_key_per_instance> tls_dble_type;

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " n" << std::endl;
		return 1;
	}

	const size_t n = atoll(argv[1]);

	double* pts;
	posix_memalign((void**) &pts, 16, sizeof(double)*n);

	tbb::task_scheduler_init init(8);

	std::cerr << "Generate a normal disitribution (mean = 0.5, standard deviation = 0.2)..." << std::endl;
#pragma omp parallel
	{
		boost::random::mt19937 gen(time(NULL));
		boost::random::normal_distribution<double> dist(0.5, 0.2);

#pragma omp for
		for (size_t i = 0; i < n; i++) {
			pts[i] = clamp(dist(gen), 0.0, 1.0);
		}
	}
	std::cerr << "Done" << std::endl;

	std::cerr << "Compute histogram..." << std::endl;
	tls_dble_type tls_hist(&init_tls);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[&tls_hist, pts](tbb::blocked_range<size_t> const& r)
		{
			double* hist_local = tls_hist.local();
			double* pts_ = pts;
			for (size_t i = r.begin(); i != r.end(); i++) {
				const size_t idx = floor(pts_[i]/SIZE_INTERVAL);
				hist_local[idx]++;
			}
		});

	if (tls_hist.size() == 0) {
		std::cerr << "Computation gave no histogram!!" << std::endl;
		return 1;
	}

	tls_dble_type::const_iterator it = tls_hist.begin();

	std::cerr << "Final reduction..." << std::endl;
	double* final_hist = *it;
	it++;
	// Final reduction
	// Note: better performance if these two loops are in the other order!
	for (; it != tls_hist.end(); it++) {
		double* hist_local = *it;
		for (size_t i = 0; i < L1_SIZE_DBLE; i++) {
			final_hist[i] += hist_local[i];
		}
	}
	std::cerr << "Done." << std::endl;

	// Compute min/max
	size_t max = 0;
	size_t min = ULLONG_MAX;

	for (size_t i = 0; i < L1_SIZE_DBLE; i++) {
		const double v = final_hist[i];
		if (v > max) {
			max = v;
		}
		if (v < min) {
			min = v;
		}
	}

	const double diff = max-min;
	char stars_80[81];
	memset(stars_80, '*', 80);
	stars_80[80] = 0;
	for (size_t i = 0; i < L1_SIZE_DBLE; i++) {
		const size_t nstars = 80.0*((double)(final_hist[i]-min)/diff);
		stars_80[nstars] = 0;
		std::cout << stars_80 << std::endl;
		stars_80[nstars] = '*';
	}

	return 0;
}
