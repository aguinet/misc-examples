#include <cstdint>
#include <iostream>
#include <string.h>
#include <limits.h>

#include "tools.h"

void display(uint32_t* hist, const uint32_t size)
{
	// Display histogram
	
	// Compute min/max
	uint32_t max = 0;
	uint32_t min = UINT_MAX;

	for (uint32_t i = 0; i < size; i++) {
		const uint32_t v = hist[i];
		if (v > max) {
			max = v;
		}
		if (v < min) {
			min = v;
		}
	}

	const uint32_t diff = max-min;
	char stars_80[81];
	memset(stars_80, '*', 80);
	stars_80[80] = 0;
	for (size_t i = 0; i < size; i++) {
		const size_t nstars = 80.0*((double)(hist[i]-min)/diff);
		stars_80[nstars] = 0;
		std::cout << stars_80 << std::endl;
		stars_80[nstars] = '*';
	}
}
