#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <iostream>
#include <cstdint>
#include <cstdlib>

#include "compute.h"
#include "tools.h"

template <typename T>
static inline T clamp(T const v, T const min, T const max)
{
	return std::max(std::min(v, max), min);
}

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cout << "Usage: " << argv[0] << " n size_interval_bytes" << std::endl;
		return 1;
	}

	const size_t n = atoll(argv[1]);
	const uint32_t size_interval_bytes = atoi(argv[2]);
	uint32_t size_interval_u32 = size_interval_bytes/sizeof(uint32_t);
	const float size_interval = 1.0f/size_interval_u32;
	
	float* pts;
	posix_memalign((void**) &pts, 16, sizeof(float)*n);

	std::cerr << "Generate a normal disitribution (mean = 0.5, standard deviation = 0.2)..." << std::endl;
	boost::random::mt19937 gen(time(NULL));
	boost::random::uniform_real_distribution<float> dist(0.0f, 0.9999f);
	for (size_t i = 0; i < n; i++) {
		pts[i] = dist(gen);
	}
	std::cerr << "Done" << std::endl;

	std::cerr << "Compute histogram..." << std::endl;

	uint32_t* hist = compute(pts, n, size_interval_u32, size_interval);

	std::cerr << "Done." << std::endl;

	if (hist == NULL) {
		std::cerr << "Histogram empty!" << std::endl;
		return 1;
	}

	//display(hist, size_interval_u32);

	return 0;
}
