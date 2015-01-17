#------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2015, The Ohio State University
# All rights reserved.
# Author: Yi Wang
#
#
#------------------------------------------------------------------------------------------------------------------------

MPI_CC = mpicxx 
CFLAGS = -Wall -O3 -std=c++11 -static-libstdc++
OPENMP_LIB = -fopenmp

PROGS = simulation

all: $(PROGS)

#Simulation Code
#To run a different application over the dummy simulation, replace the file "simulation.cpp" with any cpp file in the folder base_apps or win_apps.
simulation: simulation.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) 

clean:
	rm  -f *.o $(PROGS)
