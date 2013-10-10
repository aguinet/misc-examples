CFLAGS=-std=c99 -march=native -Ofast -Wall -ftree-vectorizer-verbose=1
LDFLAGS=-lm

all: vec minmax

vec: vec.c helpers.o
minmax: minmax.o helpers.o

clean:
	rm *.o
	rm vec
	rm minmax
