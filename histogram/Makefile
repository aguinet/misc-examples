CC=g++
CXX=g++
CXXFLAGS=-O3 -march=native -fopenmp -g -std=c++11 -Wall
LDFLAGS=-lgomp

all: main main_omp main_omp_atomic

main: main.o compute.o tools.o helpers.o
	g++ $(LDFLAGS) $^ -o $@

main_omp_atomic: main.o compute_atomic.o tools.o helpers.o
	g++ $(LDFLAGS) $^ -o $@

main_omp: main.o compute_omp.o tools.o helpers.o
	g++ $(LDFLAGS) $^ -o $@

clean:
	rm *.o
	rm main
	rm main_omp
	rm main_omp_atomic
