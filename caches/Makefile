CC=g++
CXX=g++
CXXFLAGS=-O3 -march=native -std=c++11 -fopenmp -g
LDFLAGS=-lgomp

mem_rand_bench: mem_rand_bench.o helpers.o

clean:
	rm *.o
	rm mem_rand_bench
