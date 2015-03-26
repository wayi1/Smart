#------------------------------------------------------------------------------
# Copyright (c) 2015, The Ohio State University
# All rights reserved.
# Author: Yi Wang
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of The Ohio University nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE OHIO STATE UNIVERSITY ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------ 

CC = g++
MPI_CC = mpicxx 
CFLAGS = -Wall -O3 -std=c++11 -static-libstdc++
OPENMP_LIB = -fopenmp

NETCDF_LIB = -L/usr/local/netcdf/lib -I/usr/local/netcdf/inlcude -lnetcdf 
HDF5_LIB = -L/usr/local/hdf5/lib -I/usr/local/hdf5/include -lhdf5 -lz -lhdf5_hl
FORMAT_LIB = $(NETCDF_LIB) $(HDF5_LIB)

HOME = ../..
COMMON_HEADER_DIR = examples/common_app_headers
INC_DIR = include
SRC_DIR = src

COM_OBJS = $(HOME)/$(SRC_DIR)/partitioner.o $(HOME)/$(SRC_DIR)/netcdf_partitioner.o $(HOME)/$(SRC_DIR)/hdf5_partitioner.o
PROGS = netcdf_data_generator hdf5_data_generator histogram_offline kmeans_offline logistic_regression_offline moving_avg_offline moving_median_offline savitzky_golay_offline 

all: $(PROGS)

netcdf_data_generator: netcdf_data_generator.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(NETCDF_LIB)

hdf5_data_generator: hdf5_data_generator.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(HDF5_LIB)

$(HOME)/$(SRC_DIR)/partitioner.o: $(HOME)/$(SRC_DIR)/partitioner.cpp
	$(MPI_CC) $(CFLAGS) -c $< -o $@ $(FORMAT_LIB) -I$(HOME)/$(INC_DIR)

$(HOME)/$(SRC_DIR)/netcdf_partitioner.o: $(HOME)/$(SRC_DIR)/netcdf_partitioner.cpp
	$(MPI_CC) $(CFLAGS) -c $< -o $@ $(FORMAT_LIB) -I$(HOME)/$(INC_DIR)

$(HOME)/$(SRC_DIR)/hdf5_partitioner.o: $(HOME)/$(SRC_DIR)/hdf5_partitioner.cpp
	$(MPI_CC) $(CFLAGS) -c $< -o $@ $(FORMAT_LIB) -I$(HOME)/$(INC_DIR)

histogram_offline: histogram_offline.cpp $(COM_OBJS) 
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

kmeans_offline: kmeans_offline.cpp $(COM_OBJS)
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

logistic_regression_offline: logistic_regression_offline.cpp $(COM_OBJS)
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

moving_avg_offline: moving_avg_offline.cpp $(COM_OBJS)
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

moving_median_offline: moving_median_offline.cpp $(COM_OBJS)
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

savitzky_golay_offline: savitzky_golay_offline.cpp $(COM_OBJS)
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) $(FORMAT_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR)

clean:
	rm  -f *.o $(COM_OBJS) $(PROGS)
